SYSTEM_MESSAGE = """
You are Apollo, an AI music-player assistant, designed to provide a personalized and engaging listening experience through thoughtful interaction and intelligent tool usage.

Your Main Responsibilities:

1. **Play Music:** Utilize your specialized toolkit to fulfill music requests.

2. **Mood Monitoring:** Constantly gauge the user's mood and adapt the music accordingly. For example, if the mood shifts from 'Happy' to 'more upbeat,' select 'Energetic' music. 

3. **Track and Artist Memory:** Be prepared to recall tracks and/or artists that the user has previously requested.

4. **Provide Guidance:** If the user appears indecisive or unsure about their selection, proactively offer suggestions based on their previous preferences or popular choices within the desired mood or genre.

5. **Seek Clarification:** If a user's request is ambiguous, don't hesitate to ask for more details.
"""

GENRE_LIST = [
    'acoustic', 'afrobeat', 'alt-rock', 'alternative', 'ambient', 'anime', 
    'black-metal', 'bluegrass', 'blues', 'bossanova', 'brazil', 'breakbeat', 
    'british', 'cantopop', 'chicago-house', 'children', 'chill', 'classical', 
    'club', 'comedy', 'country', 'dance', 'dancehall', 'death-metal', 
    'deep-house', 'detroit-techno', 'disco', 'disney', 'drum-and-bass', 'dub', 
    'dubstep', 'edm', 'electro', 'electronic', 'emo', 'folk', 'forro', 'french', 
    'funk', 'garage', 'german', 'gospel', 'goth', 'grindcore', 'groove', 
    'grunge', 'guitar', 'happy', 'hard-rock', 'hardcore', 'hardstyle', 
    'heavy-metal', 'hip-hop', 'holidays', 'honky-tonk', 'house', 'idm', 
    'indian', 'indie', 'indie-pop', 'industrial', 'iranian', 'j-dance', 
    'j-idol', 'j-pop', 'j-rock', 'jazz', 'k-pop', 'kids', 'latin', 'latino', 
    'malay', 'mandopop', 'metal', 'metal-misc', 'metalcore', 'minimal-techno', 
    'movies', 'mpb', 'new-age', 'new-release', 'opera', 'pagode', 'party', 
    'philippines-opm', 'piano', 'pop', 'pop-film', 'post-dubstep', 'power-pop', 
    'progressive-house', 'psych-rock', 'punk', 'punk-rock', 'r-n-b', 
    'rainy-day', 'reggae', 'reggaeton', 'road-trip', 'rock', 'rock-n-roll', 
    'rockabilly', 'romance', 'sad', 'salsa', 'samba', 'sertanejo', 'show-tunes', 
    'singer-songwriter', 'ska', 'sleep', 'songwriter', 'soul', 'soundtracks', 
    'spanish', 'study', 'summer', 'swedish', 'synth-pop', 'tango', 'techno', 
    'trance', 'trip-hop', 'turkish', 'work-out', 'world-music'
]


HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <link href="https://fonts.googleapis.com/css2?family=Gruppo&display=swap" rel="stylesheet">
  <style>
    h1 {
      font-family: 'Gruppo', sans-serif;
    }
    .hover-item:hover {
      color: white !important;
    }
    .hover-item a:hover {
      font-size: 105%;
    }
    a {
      color: inherit !important;
      text-decoration: none !important;
    }
    .container {
      position: relative;
    }
    .apollo-gif {
      position: absolute;
      top: 60%;
      right: 9%;
      transform: translateY(-50%);
    }
  </style>
</head>
<body>
  <div class="container" style="text-align: left; margin-top: 50px;">
    <h1 style="color: white; font-size: 56px;">Apollo<sup style="color: #E457E2;">AI</sup> Music Assistant</h1>
    <h2 style="color: #E457E2; font-size: 26px;">Experience personalized, intelligent music interaction like never before</h2>
    <ul style="list-style-type: none; padding: 12px 0; margin: 0;">
      <li class="hover-item" style="color: #CECECE; font-size: 26px; padding: 12px 0;"><span style="color: #5491FA;">1.</span> Ensure your Spotify client is open</li>
      <li class="hover-item" style="color: #CECECE; font-size: 26px; padding: 12px 0;"><span style="color: #5491FA;">2.</span> Login to <a href="https://developer.spotify.com">Spotify Developer</a> website</li>
      <li class="hover-item" style="color: #CECECE; font-size: 26px; padding: 12px 0;"><span style="color: #5491FA;">3.</span> Go to <a href="https://developer.spotify.com/dashboard">Dashboard</a> then 'Create app'</li> 
      <li class="hover-item" style="color: #CECECE; font-size: 26px; padding: 12px 0; margin-bottom: -12px;"><span style="color: #5491FA;">4.</span> Set Redirect URI to <a href="https://jonaswaller.com">https://jonaswaller.com</a></li>
      <li style="color: #969696; font-size: 20px; padding: 0; margin-top: -12px; text-indent: 55px;">The other fields can be anything</li>
      <li class="hover-item" style="color: #CECECE; font-size: 26px; padding: 12px 0;"><span style="color: #5491FA;">5.</span> Hit 'Save' then 'Settings'</li>
      <li class="hover-item" style="color: #CECECE; font-size: 26px; padding: 12px 0;"><span style="color: #5491FA;">6.</span> Copy your Spotify Client ID</li>
    </ul>
    <img src="file/apollo.gif" class="apollo-gif" width='400'>
  </div>
</body>
</html>
"""


APOLLO_MESSAGE = """
Welcome! Tell me your **mood** to help me select the best music for you

### <span style="color: #E457E2;"> 💿 Standard 💿</span>
- **Specific Song:** Play Passionfruit 
- **Controls:** Queue, Pause, Resume, Skip
- **More Info:** Explain this song 
- **Album:** Play the album from Barbie
- **Playlist:** Play my Shower playlist

### <span style="color: #E457E2;"> 💿 Mood-Based 💿</span>
- **Genre:** I'm happy, play country 
- **Artist:** Play Migos hype songs
- **Recommendations:** I love Drake and house, recommend songs
- **Create Playlist:** Make a relaxing playlist of SZA-like songs
"""
