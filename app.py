# Standard Library Imports
import os
import random
import re
import time
from urllib.parse import urlparse, parse_qs

# Third-Party Imports
import gradio as gr
import lyricsgenius
import requests
import spotipy
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fuzzywuzzy import fuzz
from pydantic import BaseModel, Field
from requests.exceptions import Timeout
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from spotipy.exceptions import SpotifyException

# Local Application/Library Specific Imports
import openai
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor, tool
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage


from messages import SYSTEM_MESSAGE, GENRE_LIST, HTML, APOLLO_MESSAGE
from dotenv import load_dotenv
load_dotenv()  


# ------------------------------
# Section: Global Vars
# ------------------------------


GENIUS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DEBUG_MODE = True 
def debug_print(*args, **kwargs):
    if DEBUG_MODE:
        print(*args, **kwargs)
    
THEME = gr.themes.Default(
    primary_hue=gr.themes.colors.blue,
    font=[gr.themes.GoogleFont("Space Mono"), "monospace", "sans-serif"],
    spacing_size=gr.themes.sizes.spacing_sm,
    radius_size=gr.themes.sizes.radius_sm,
    text_size=gr.themes.sizes.text_lg
).set(
    body_background_fill="#000000", 
    #button_primary_background_fill="white",
    #button_primary_text_color="black",
    button_primary_background_fill_hover="black",
    #button_primary_text_color_hover="white"

)

# TODO: switch to personal website
REDIRECT_URI = "https://jonaswaller.com"

# Spotify functions
SCOPE = [
    'user-library-read',
    'user-read-playback-state',
    'user-modify-playback-state',
    'playlist-modify-public',
    'user-top-read'
]

MOOD_SETTINGS = {
    "happy": {"max_instrumentalness": 0.001, "min_valence": 0.6},
    "sad": {"max_danceability": 0.65, "max_valence": 0.4},
    "energetic": {"min_tempo": 120, "min_danceability": 0.75},
    "calm": {"max_energy": 0.65, "max_tempo": 130}
}

# genre + mood function
NUM_ARTISTS = 20 # artists to retrieve from user's top artists
TIME_RANGE = "medium_term" # short, medium, long
NUM_TRACKS = 10 # tracks to add to playback
MAX_ARTISTS = 4 # sp.recommendations() seeds: 4/5 artists, 1/5 genre

# artist + mood function
NUM_ALBUMS = 20 # maximum number of albums to retrieve from an artist
MAX_TRACKS = 10 # tracks to randomly select from an artist

# matching playlists + moods
MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') # smaller BERT
os.environ["TOKENIZERS_PARALLELISM"] = "false" # warning
MOOD_LIST = ["happy", "sad", "energetic", "calm"]
MOOD_EMBEDDINGS = MODEL.encode(MOOD_LIST)
GENRE_EMBEDDINGS = MODEL.encode(GENRE_LIST) 

# agent tools
RETURN_DIRECT = True 

#LLM_MODEL = "gpt-3.5-turbo-0613"
LLM_MODEL = "gpt-4"

# adjectives for playlist names
THEMES = ["Epic", "Hypnotic", "Dreamy", "Legendary", "Majestic", 
          "Enchanting", "Ethereal", "Super Lit", "Harmonious", "Heroic"]


with gr.Blocks(theme=THEME) as app:

    # ------------------------------
    # Section: Spotify Authentication 
    # ------------------------------
    gr.HTML(HTML)

    # incredibly important for preserving using isolation
    ACCESS_TOKEN_VAR = gr.State() 
    AGENT_EXECUTOR_VAR = gr.State()

    with gr.Row():
        client_id = gr.Textbox(placeholder="7. Paste Spotify Client ID", container=False, text_align="left")
        generate_link = gr.Button("Submit ID", variant="primary")
    display_link = gr.HTML()
    
    with gr.Row():
        url = gr.Textbox(placeholder="9. Paste entire URL", container=False, text_align="left")
        authorize_url = gr.Button("Submit URL", variant="primary") 
    auth_result = gr.HTML()


    def spotify_auth(client_id, url=None, access_tokens=None):
        """
        Authenticate Spotify with the provided client_id and url.
        """
        if url:
            parsed_url = urlparse(url)
            fragment = parsed_url.fragment
            access_token = parse_qs(fragment)['access_token'][0]
            debug_print(access_token)

            return access_token, '<p class="hover-item" style="color: #CECECE; font-size: 26px; padding: 12px 0; text-align: left;">Your Spotify is connected!</p>'

        else:
            auth_url = (
                f"https://accounts.spotify.com/authorize?response_type=token&client_id={client_id}"
                f"&scope={'%20'.join(SCOPE)}&redirect_uri={REDIRECT_URI}"
            )
            return f'<p class="hover-item" style="color: #CECECE; font-size: 26px; padding: 12px 0;">' \
                f'<span style="color: #5491FA;">8.</span> Click <a href="{auth_url}" target="_blank" ' \
                f'style="color: inherit !important; text-decoration: none !important;" ' \
                f'onmouseover="this.style.fontSize=\'105%\';" onmouseout="this.style.fontSize=\'100%\';">' \
                f'here</a> and copy the entire URL</p>'


    generate_link.click(spotify_auth, inputs=[client_id], outputs=display_link)
    authorize_url.click(spotify_auth, inputs=[client_id, url, ACCESS_TOKEN_VAR], outputs=[ACCESS_TOKEN_VAR, auth_result]) 

    create_agent_button = gr.Button("Start Session", variant="primary")
    create_agent_result = gr.HTML()

    def create_agent(access_token):

        # included 'client' parameter to fix error
        llm = ChatOpenAI(
            model=LLM_MODEL, 
            openai_api_key=OPENAI_API_KEY, 
            streaming=True, 
            callbacks=[StreamingStdOutCallbackHandler()], 
            client=openai.ChatCompletion
        )


        # ------------------------------
        # Section: Spotify Functions 
        # ------------------------------


        sp = spotipy.Spotify(auth=access_token)
        devices = sp.devices()
        device_id = devices['devices'][0]['id']


        def find_track_by_name(track_name):
            """
            Finds the Spotify track URI given the track name.
            """
            results = sp.search(q=track_name, type='track')
            track_uri = results['tracks']['items'][0]['uri']
            return track_uri


        def play_track_by_name(track_name): 
            """
            Plays a track given its name. Uses the above function.
            """
            track_uri = find_track_by_name(track_name)
            track_name = sp.track(track_uri)["name"]
            artist_name = sp.track(track_uri)['artists'][0]['name']

            try:
                sp.start_playback(device_id=device_id, uris=[track_uri])
                return f"♫ Now playing {track_name} by {artist_name} ♫"
            except SpotifyException as e:
                return f"An error occurred with Spotify: {e}. \n\n**Remember to wake up Spotify.**"
            except Exception as e: 
                return f"An unexpected error occurred: {e}."
            

        def queue_track_by_name(track_name):
            """
            Queues track given its name.
            """
            track_uri = find_track_by_name(track_name)
            track_name = sp.track(track_uri)["name"]
            sp.add_to_queue(uri=track_uri, device_id=device_id)
            return f"♫ Added {track_name} to your queue ♫"
        

        def pause_track():
            """
            Pauses the current playback.
            """
            sp.pause_playback(device_id=device_id)
            return "♫ Playback paused ♫"


        def resume_track():
            """
            Resumes the current playback.
            """
            sp.start_playback(device_id=device_id)
            return "♫ Playback started ♫"


        def skip_track():
            """
            Skips the current playback.
            """
            sp.next_track(device_id=device_id)
            return "♫ Skipped to your next track ♫"
        
            
        ### ### ### More Elaborate Functions ### ### ###
            

        def play_album_by_name_and_artist(album_name, artist_name):
            """
            Plays an album given its name and the artist.
            context_uri (provide a context_uri to start playback of an album, artist, or playlist) expects a string.
            """  
            results = sp.search(q=f'{album_name} {artist_name}', type='album')
            album_id = results['albums']['items'][0]['id']
            album_info = sp.album(album_id)
            album_name = album_info['name']
            artist_name = album_info['artists'][0]['name']

            try: 
                sp.start_playback(device_id=device_id, context_uri=f'spotify:album:{album_id}')
                return f"♫ Now playing {album_name} by {artist_name} ♫"
            except spotipy.SpotifyException as e:
                return f"An error occurred with Spotify: {e}. \n\n**Remember to wake up Spotify.**"
            except Timeout:
                return f"An unexpected error occurred: {e}."


        def play_playlist_by_name(playlist_name):
            """
            Plays an existing playlist in the user's library given its name.
            """
            playlists = sp.current_user_playlists()
            playlist_dict = {playlist['name']: (playlist['id'], playlist['owner']['display_name']) for playlist in playlists['items']}
            playlist_names = [key for key in playlist_dict.keys()]

            # defined inside to capture user-specific playlists
            playlist_name_embeddings = MODEL.encode(playlist_names)
            user_playlist_embedding = MODEL.encode([playlist_name])

            # compares (embedded) given name to (embedded) playlist library and outputs the closest match
            similarity_scores = cosine_similarity(user_playlist_embedding, playlist_name_embeddings)
            most_similar_index = similarity_scores.argmax()
            playlist_name = playlist_names[most_similar_index]

            try: 
                playlist_id, creator_name = playlist_dict[playlist_name]
                sp.start_playback(device_id=device_id, context_uri=f'spotify:playlist:{playlist_id}')
                return f'♫ Now playing {playlist_name} by {creator_name} ♫'
            except: 
                return "Unable to find playlist. Please try again."


        def get_track_info(): 
            """
            Harvests information for explain_track() using Genius' API and basic webscraping. 
            """
            current_track_item = sp.current_user_playing_track()['item']
            track_name = current_track_item['name']
            artist_name = current_track_item['artists'][0]['name']
            album_name = current_track_item['album']['name']
            release_date = current_track_item['album']['release_date']
            basic_info = {
                'track_name': track_name,
                'artist_name': artist_name,
                'album_name': album_name,
                'release_date': release_date,
            }

            # define inside to avoid user conflicts (simultaneously query Genius)
            genius = lyricsgenius.Genius(GENIUS_TOKEN)
            # removing feature information from song titles to avoid scewing search
            track_name = re.split(' \(with | \(feat\. ', track_name)[0]
            result = genius.search_song(track_name, artist_name)

            # if no Genius page exists
            if result is not None and hasattr(result, 'artist'):
                genius_artist = result.artist.lower().replace(" ", "")
                spotify_artist = artist_name.lower().replace(" ", "")
                debug_print(spotify_artist)
                debug_print(genius_artist)
                if spotify_artist not in genius_artist:
                    return basic_info, None, None, None
            else: 
                genius_artist = None
                return basic_info, None, None, None
            
            # if Genius page exists
            lyrics = result.lyrics
            url = result.url
            response = requests.get(url)

            # parsing the webpage and locating 'About' section
            soup = BeautifulSoup(response.text, 'html.parser')
            # universal 'About' section element across all Genius song lyrics pages
            about_section = soup.select_one('div[class^="RichText__Container-oz284w-0"]')

            # if no 'About' section exists
            if not about_section: 
                return basic_info, None, lyrics, url
            
            # if 'About' section exists
            else: 
                about_section = about_section.get_text(separator='\n')
                return basic_info, about_section, lyrics, url


        def explain_track(): 
            """
            Displays track information in an organized, informational, and compelling manner. 
            Uses the above function.
            """
            basic_info, about_section, lyrics, url = get_track_info()
            debug_print(basic_info, about_section, lyrics, url)

            if lyrics: # if Genius page exists
                system_message_content = """
                    Your task is to create an engaging summary for a track using the available details
                    about the track and its lyrics. If there's insufficient or no additional information
                    besides the lyrics, craft the entire summary based solely on the lyrical content."
                    """
                human_message_content = f"{about_section}\n\n{lyrics}"
                messages = [
                    SystemMessage(content=system_message_content),
                    HumanMessage(content=human_message_content)
                ]
                ai_response = llm(messages).content
                summary = f"""
                    **Name:** <span style="color: #E457E2; font-weight: bold; font-style: italic;">{basic_info["track_name"]}</span>   
                    **Artist:** {basic_info["artist_name"]}   
                    **Album:** {basic_info["album_name"]}   
                    **Release:** {basic_info["release_date"]}   

                    **About:** 
                    {ai_response}

                    <a href='{url}'>Click here for more information on Genius!</a>  
                """
                return summary
            
            else: # if no Genius page exists
                url = "https://genius.com/Genius-how-to-add-songs-to-genius-annotated"
                summary = f"""
                    **Name:** <span style="color: #E457E2; font-weight: bold; font-style: italic;">{basic_info["track_name"]}</span>   
                    **Artist:** {basic_info["artist_name"]}   
                    **Album:** {basic_info["album_name"]}   
                    **Release:** {basic_info["release_date"]}   

                    **About:** 
                    Unfortunately, this track has not been uploaded to Genius.com

                    <a href='{url}'>Be the first to change that!</a>  
                """
                return summary


        ### ### ### Genre + Mood ### ### ###


        def get_user_mood(user_mood):
            """
            Categorizes the user's mood as either 'happy', 'sad', 'energetic', or 'calm'.
            Uses same cosine similarity/embedding concepts as with determining playlist names.
            """
            if user_mood.lower() in MOOD_LIST:
                user_mood = user_mood.lower()
                return user_mood
            else:
                user_mood_embedding = MODEL.encode([user_mood.lower()])
                similarity_scores = cosine_similarity(user_mood_embedding, MOOD_EMBEDDINGS)
                most_similar_index = similarity_scores.argmax()
                user_mood = MOOD_LIST[most_similar_index]
                return user_mood


        def get_genre_by_name(genre_name): 
            """
            Matches user's desired genre to closest (most similar) existing genre in the list of genres.
            recommendations() only accepts genres from this list.
            """
            if genre_name.lower() in GENRE_LIST:
                genre_name = genre_name.lower()
                return genre_name
            else:
                genre_name_embedding = MODEL.encode([genre_name.lower()])
                similarity_scores = cosine_similarity(genre_name_embedding, GENRE_EMBEDDINGS)
                most_similar_index = similarity_scores.argmax()
                genre_name = GENRE_LIST[most_similar_index]
                return genre_name


        def is_genre_match(genre1, genre2, threshold=75):
            """
            Determines if two genres are semantically similar.
            token_set_ratio() - for quantifying semantic similarity - and 
            threshold of 75 (out of 100) were were arbitrarily determined through basic testing.
            """
            score = fuzz.token_set_ratio(genre1, genre2)
            debug_print(score) 
            return score >= threshold


        def create_track_list_str(track_uris):
            """
            Creates an organized list of track names. 
            Used in final return statements by functions below.
            """
            track_details = sp.tracks(track_uris)
            track_names_with_artists = [f"{track['name']} by {track['artists'][0]['name']}" for track in track_details['tracks']]
            track_list_str = "<br>".join(track_names_with_artists) 
            return track_list_str


        def play_genre_by_name_and_mood(genre_name, user_mood):
            """
            1. Retrieves user's desired genre and current mood.
            2. Matches genre and mood to existing options.
            3. Gets 4 of user's top artists that align with genre.
            4. Conducts personalized recommendations() search.
            5. Plays selected track, clears the queue, and adds the rest to the now-empty queue.
            """
            genre_name = get_genre_by_name(genre_name)
            user_mood = get_user_mood(user_mood).lower()
            debug_print(genre_name) 
            debug_print(user_mood)
            
            # increased personalization
            user_top_artists = sp.current_user_top_artists(limit=NUM_ARTISTS, time_range=TIME_RANGE) 
            matching_artists_ids = []

            for artist in user_top_artists['items']:
                debug_print(artist['genres']) 
                for artist_genre in artist['genres']:
                    if is_genre_match(genre_name, artist_genre):
                        matching_artists_ids.append(artist['id'])
                        break # don't waste time cycling artist genres after match
                if len(matching_artists_ids) == MAX_ARTISTS:
                    break 

            if not matching_artists_ids:
                matching_artists_ids = None
            else: 
                artist_names = [artist['name'] for artist in sp.artists(matching_artists_ids)['artists']]
                debug_print(artist_names)
                debug_print(matching_artists_ids)

            recommendations = sp.recommendations( # accepts maximum {genre + artists} = 5 seeds
                                            seed_artists=matching_artists_ids, 
                                            seed_genres=[genre_name], 
                                            seed_tracks=None, 
                                            limit=NUM_TRACKS, # number of tracks to return
                                            country=None,
                                            **MOOD_SETTINGS[user_mood]) # maps to mood settings dictionary
                                            
            track_uris = [track['uri'] for track in recommendations['tracks']]
            track_list_str = create_track_list_str(track_uris)
            sp.start_playback(device_id=device_id, uris=track_uris)

            return f"""
            **♫ Now Playing:** <span style="color: #E457E2; font-weight: bold; font-style: italic;">{genre_name}</span> ♫

            **Selected Tracks:**\n
            {track_list_str}
            """


        ### ### ### Artist + Mood ### ### ###


        def play_artist_by_name_and_mood(artist_name, user_mood):
            """
            Plays tracks (randomly selected) by a given artist that matches the user's mood.
            """
            user_mood = get_user_mood(user_mood).lower()
            debug_print(user_mood)

            # retrieving and shuffling all artist's tracks
            first_name = artist_name.split(',')[0].strip()
            results = sp.search(q=first_name, type='artist')
            artist_id = results['artists']['items'][0]['id']
            # most recent albums retrieved first
            artist_albums = sp.artist_albums(artist_id, album_type='album', limit=NUM_ALBUMS)
            artist_tracks = []
            for album in artist_albums['items']:
                album_tracks = sp.album_tracks(album['id'])['items']
                artist_tracks.extend(album_tracks)
            random.shuffle(artist_tracks) 

            # filtering until we find enough (MAX_TRACKS) tracks that match user's mood
            selected_tracks = []
            for track in artist_tracks:
                if len(selected_tracks) == MAX_TRACKS: 
                    break
                features = sp.audio_features([track['id']])[0]
                mood_criteria = MOOD_SETTINGS[user_mood]

                match = True
                for criteria, threshold in mood_criteria.items():
                    if "min_" in criteria and features[criteria[4:]] < threshold:
                        match = False
                        break
                    elif "max_" in criteria and features[criteria[4:]] > threshold:
                        match = False
                        break
                if match:
                    debug_print(f"{features}\n{mood_criteria}\n\n") 
                    selected_tracks.append(track)

            track_names = [track['name'] for track in selected_tracks]
            track_list_str = "<br>".join(track_names)  # using HTML line breaks for each track name
            debug_print(track_list_str)
            track_uris = [track['uri'] for track in selected_tracks]
            sp.start_playback(device_id=device_id, uris=track_uris)

            return f"""
            **♫ Now Playing:** <span style="color: #E457E2; font-weight: bold; font-style: italic;">{artist_name}</span> ♫

            **Selected Tracks:**\n
            {track_list_str}
            """


        ### ### ### Recommendations ### ### ###


        def recommend_tracks(genre_name=None, artist_name=None, track_name=None, user_mood=None):
            """
            1. Retrieves user's preferences based on artist_name, track_name, genre_name, and/or user_mood.
            2. Uses these parameters to conduct personalized recommendations() search.
            3. Returns the track URIs of (NUM_TRACKS) recommendation tracks.
            """
            user_mood = get_user_mood(user_mood).lower() if user_mood else None
            debug_print(user_mood)

            seed_genre, seed_artist, seed_track = None, None, None

            if genre_name:
                first_name = genre_name.split(',')[0].strip() 
                genre_name = get_genre_by_name(first_name)
                seed_genre = [genre_name]
                debug_print(seed_genre)

            if artist_name:
                first_name = artist_name.split(',')[0].strip() # if user provides multiple artists, use the first
                results = sp.search(q='artist:' + first_name, type='artist')
                seed_artist = [results['artists']['items'][0]['id']]

            if track_name:
                first_name = track_name.split(',')[0].strip()
                results = sp.search(q='track:' + first_name, type='track')
                seed_track = [results['tracks']['items'][0]['id']]
            
            # if user requests recommendations without specifying anything but their mood
            # this is because recommendations() requires at least one seed
            if seed_genre is None and seed_artist is None and seed_track is None:
                raise ValueError("At least one genre, artist, or track must be provided.")
            
            recommendations = sp.recommendations( # passing in 3 seeds
                                    seed_artists=seed_artist, 
                                    seed_genres=seed_genre, 
                                    seed_tracks=seed_track, 
                                    limit=NUM_TRACKS,
                                    country=None,
                                    **MOOD_SETTINGS[user_mood] if user_mood else {})
            
            track_uris = [track['uri'] for track in recommendations['tracks']]
            return track_uris


        def play_recommended_tracks(genre_name=None, artist_name=None, track_name=None, user_mood=None):
            """
            Plays the track_uris returned by recommend_tracks().
            """
            try:
                track_uris = recommend_tracks(genre_name, artist_name, track_name, user_mood)
                track_list_str = create_track_list_str(track_uris) 
                sp.start_playback(device_id=device_id, uris=track_uris)

                return f"""
                **♫ Now Playing Recommendations Based On:** <span style="color: #E457E2; font-weight: bold; font-style: italic;">
                {', '.join(filter(None, [genre_name, artist_name, track_name, "Your Mood"]))}</span> ♫

                **Selected Tracks:**\n
                {track_list_str}
                """
            except ValueError as e:
                return str(e)  


        def create_playlist_from_recommendations(genre_name=None, artist_name=None, track_name=None, user_mood=None):
            """
            Creates a playlist from recommend_tracks(). 
            """
            user = sp.current_user()
            user_id = user['id']
            user_name = user["display_name"]

            playlists = sp.current_user_playlists()
            playlist_names = [playlist['name'] for playlist in playlists["items"]]
            chosen_theme = random.choice(THEMES)
            playlist_name = f"{user_name}'s {chosen_theme} Playlist"
            # ensuring the use of new adjective each time
            while playlist_name in playlist_names:
                chosen_theme = random.choice(THEMES)
                playlist_name = f"{user_name}'s {chosen_theme} Playlist"

            playlist_description=f"Apollo AI's personalized playlist for {user_name}. Get yours here: (add link)." # TODO: add link to project
            new_playlist = sp.user_playlist_create(user=user_id, name=playlist_name, 
                                                            public=True, collaborative=False, description=playlist_description)

            track_uris = recommend_tracks(genre_name, artist_name, track_name, user_mood)
            track_list_str = create_track_list_str(track_uris) 
            sp.user_playlist_add_tracks(user=user_id, playlist_id=new_playlist['id'], tracks=track_uris, position=None)
            playlist_url = f"https://open.spotify.com/playlist/{new_playlist['id']}"

            return f"""
            ♫ Created *{playlist_name}* Based On: <span style='color: #E457E2; font-weight: bold; font-style: italic;'>
            {', '.join(filter(None, [genre_name, artist_name, track_name, 'Your Mood']))}</span> ♫

            **Selected Tracks:**\n
            {track_list_str}

            <a href='{playlist_url}'>Click here to listen to the playlist on Spotify!</a>
            """
        

        # ------------------------------
        # Section: Agent Tools
        # ------------------------------


        class TrackNameInput(BaseModel):
            track_name: str = Field(description="Track name in the user's request.")  


        class AlbumNameAndArtistNameInput(BaseModel):
            album_name: str = Field(description="Album name in the user's request.")
            artist_name: str = Field(description="Artist name in the user's request.") 


        class PlaylistNameInput(BaseModel):
            playlist_name: str = Field(description="Playlist name in the user's request.") 


        class GenreNameAndUserMoodInput(BaseModel):
            genre_name: str = Field(description="Genre name in the user's request.")
            user_mood: str = Field(description="User's current mood/state-of-being.") 


        class ArtistNameAndUserMoodInput(BaseModel):
            artist_name: str = Field(description="Artist name in the user's request.") 
            user_mood: str = Field(description="User's current mood/state-of-being.") 


        class RecommendationsInput(BaseModel):
            genre_name: str = Field(description="Genre name in the user's request.")
            artist_name: str = Field(description="Artist name in the user's request.")
            track_name: str = Field(description="Track name in the user's request.")
            user_mood: str = Field(description="User's current mood/state-of-being.") 


        @tool("play_track_by_name", return_direct=RETURN_DIRECT, args_schema=TrackNameInput) 
        def tool_play_track_by_name(track_name: str) -> str:
            """
            Use this tool when a user wants to play a particular track by its name. 
            You will need to identify the track name from the user's request. 
            Usually, the requests will look like 'play {track name}'. 
            This tool is specifically designed for clear and accurate track requests.
            """
            return play_track_by_name(track_name)


        @tool("queue_track_by_name", return_direct=RETURN_DIRECT, args_schema=TrackNameInput)
        def tool_queue_track_by_name(track_name: str) -> str:
            """
            Always use this tool when a user says "queue" in their request.
            """
            return queue_track_by_name(track_name)


        @tool("pause_track", return_direct=RETURN_DIRECT)
        def tool_pause_track(query: str) -> str:
            """
            Always use this tool when a user says "pause" or "stop" in their request.
            """
            return pause_track()


        @tool("resume_track", return_direct=RETURN_DIRECT) 
        def tool_resume_track(query: str) -> str:
            """
            Always use this tool when a user says "resume" or "unpause" in their request.
            """
            return resume_track()


        @tool("skip_track", return_direct=RETURN_DIRECT)
        def tool_skip_track(query: str) -> str:
            """
            Always use this tool when a user says "skip" or "next" in their request.
            """
            return skip_track()


        @tool("play_album_by_name_and_artist", return_direct=RETURN_DIRECT, args_schema=AlbumNameAndArtistNameInput) 
        def tool_play_album_by_name_and_artist(album_name: str, artist_name: str) -> str:
            """
            Use this tool when a user wants to play an album.
            You will need to identify both the album name and artist name from the user's request.
            Usually, the requests will look like 'play the album {album_name} by {artist_name}'. 
            """
            return play_album_by_name_and_artist(album_name, artist_name)


        @tool("play_playlist_by_name", return_direct=RETURN_DIRECT, args_schema=PlaylistNameInput) 
        def tool_play_playlist_by_name(playlist_name: str) -> str:
            """
            Use this tool when a user wants to play one of their playlists.
            You will need to identify the playlist name from the user's request. 
            """
            return play_playlist_by_name(playlist_name)


        @tool("explain_track", return_direct=RETURN_DIRECT) 
        def tool_explain_track(query: str) -> str:
            """
            Use this tool when a user wants to know about the currently playing track.
            """
            return explain_track()


        @tool("play_genre_by_name_and_mood", return_direct=RETURN_DIRECT, args_schema=GenreNameAndUserMoodInput) 
        def tool_play_genre_by_name_and_mood(genre_name: str, user_mood: str) -> str:
            """
            Use this tool when a user wants to play a genre.
            You will need to identify both the genre name from the user's request, 
            and also their current mood, which you should always be monitoring. 
            """
            return play_genre_by_name_and_mood(genre_name, user_mood)


        @tool("play_artist_by_name_and_mood", return_direct=RETURN_DIRECT, args_schema=ArtistNameAndUserMoodInput) 
        def tool_play_artist_by_name_and_mood(artist_name: str, user_mood: str) -> str:
            """
            Use this tool when a user wants to play an artist.
            You will need to identify both the artist name from the user's request, 
            and also their current mood, which you should always be monitoring. 
            If you don't know the user's mood, ask them before using this tool.
            """
            return play_artist_by_name_and_mood(artist_name, user_mood)


        @tool("play_recommended_tracks", return_direct=RETURN_DIRECT, args_schema=RecommendationsInput) 
        def tool_play_recommended_tracks(genre_name: str, artist_name: str, track_name: str, user_mood: str) -> str:
            """
            Use this tool when a user wants track recommendations.
            You will need to identify the genre name, artist name, and/or track name
            from the user's request... and also their current mood, which you should always be monitoring.
            The user must provide at least genre, artist, or track.
            """
            return play_recommended_tracks(genre_name, artist_name, track_name, user_mood)


        @tool("create_playlist_from_recommendations", return_direct=RETURN_DIRECT, args_schema=RecommendationsInput) 
        def tool_create_playlist_from_recommendations(genre_name: str, artist_name: str, track_name: str, user_mood: str) -> str:
            """
            Use this tool when a user wants a playlist created (from recommended tracks).
            You will need to identify the genre name, artist name, and/or track name
            from the user's request... and also their current mood, which you should always be monitoring.
            The user must provide at least genre, artist, or track.
            """
            return create_playlist_from_recommendations(genre_name, artist_name, track_name, user_mood)


        CUSTOM_TOOLS =[
            tool_play_track_by_name,
            tool_queue_track_by_name,
            tool_pause_track,
            tool_resume_track,
            tool_skip_track,
            tool_play_album_by_name_and_artist,
            tool_play_playlist_by_name,
            tool_explain_track,
            tool_play_genre_by_name_and_mood,
            tool_play_artist_by_name_and_mood,
            tool_play_recommended_tracks,
            tool_create_playlist_from_recommendations
        ]


        # ------------------------------
        # Section: Chatbot
        # ------------------------------


        system_message = SystemMessage(content=SYSTEM_MESSAGE)
        MEMORY_KEY = "chat_history"
        prompt = OpenAIFunctionsAgent.create_prompt(
            system_message=system_message,
            extra_prompt_messages=[MessagesPlaceholder(variable_name=MEMORY_KEY)]
        )
        memory = ConversationBufferMemory(memory_key=MEMORY_KEY, return_messages=True)
        # NOTE: llm defined above to power explain_track() function
        agent = OpenAIFunctionsAgent(llm=llm, tools=CUSTOM_TOOLS, prompt=prompt)
        agent_executor = AgentExecutor(agent=agent, tools=CUSTOM_TOOLS, memory=memory, verbose=True)

        return agent_executor, '<p class="hover-item" style="color: #CECECE; font-size: 26px; padding: 12px 0; text-align: left;">Success! Type -music to view commands</p>'
    
    create_agent_button.click(create_agent, inputs=[ACCESS_TOKEN_VAR], outputs=[AGENT_EXECUTOR_VAR, create_agent_result])


    # ------------------------------
    # Section: Chat Interface
    # ------------------------------


    chatbot = gr.Chatbot(
        bubble_full_width=False, 
        label="Apollo",
        height=460,
        avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.png")))
    )
    msg = gr.Textbox(
        placeholder="What would you like to hear?", 
        container=False, 
        text_align="left"
    )

    def respond(user_message, chat_history, agent_executor):
        try:
            if user_message.strip() == "-music":
                bot_message = APOLLO_MESSAGE
            else:
                bot_message = agent_executor.run(user_message)
            chat_history.append((user_message, bot_message))
        except Exception as e:
            bot_message = "Error: Unable to **connect to your Spotify** | Please ensure you followed the above steps correctly"
            chat_history.append((user_message, bot_message))
        time.sleep(1)
        return "", chat_history
    
    msg.submit(respond, inputs=[msg, chatbot, AGENT_EXECUTOR_VAR], outputs=[msg, chatbot])

    gr.Examples(["Play chill rap", 
                "I'm feeling great, match my vibe", 
                "Make me a relaxing playlist of SZA-like songs"], 
                inputs=[msg], label="Quick Start 🚀")
    
    gr.HTML('''
    <p class="hover-item" style="color: #CECECE; font-size: 13px; padding: 12px 0; text-align: left;">
        <a href="YOUR_SOURCE_CODE_URL" target="_blank">GitHub Repo</a> | 
        I'd love to hear your feedback: stuart.j.waller@vanderbilt.edu
    </p>
    ''')
app.launch()
#app.launch(share=True)







