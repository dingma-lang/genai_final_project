"""Reference lists for masking Guardian match previews.

Edit these lists to change what is masked; mask_previews.py reads them. Matching is
case-sensitive and on whole words. Longer names win over shorter ones that overlap
("Villa Park" before "Villa", "Hull City" before "City").
"""

# ---------------------------------------------------------------------------
# Clubs in the analysis data (football-data spelling) -> every name used for them.
# Generic nicknames shared by several clubs (City, United, Blues, Reds, Albion, Town,
# Rangers) are listed under every club that uses them; mask_previews.py assigns them
# to HOME or AWAY only when exactly one of the two teams uses that nickname.
# ---------------------------------------------------------------------------
CLUB_ALIASES = {
    "Arsenal": ["Arsenal", "Gunners"],
    "Aston Villa": ["Aston Villa", "Villa", "Villans"],
    "Bournemouth": ["AFC Bournemouth", "Bournemouth", "Cherries"],
    "Brighton": ["Brighton & Hove Albion", "Brighton and Hove Albion", "Brighton", "Seagulls", "Albion"],
    "Burnley": ["Burnley", "Clarets"],
    "Cardiff": ["Cardiff City", "Cardiff", "Bluebirds", "City"],
    "Chelsea": ["Chelsea", "Blues"],
    "Crystal Palace": ["Crystal Palace", "Palace", "Eagles"],
    "Everton": ["Everton", "Toffees", "Blues"],
    "Fulham": ["Fulham", "Cottagers"],
    "Huddersfield": ["Huddersfield Town", "Huddersfield", "Terriers", "Town"],
    "Hull": ["Hull City", "Hull", "Tigers", "City"],
    "Leicester": ["Leicester City", "Leicester", "Foxes", "City"],
    "Liverpool": ["Liverpool", "Reds"],
    "Man City": ["Manchester City", "Manchester city", "Man City", "Citizens", "City", "Blues"],
    "Man United": ["Manchesterchester United",  # typo in one Guardian preview
                   "Manchester United", "Manchester Utd", "Man United", "Man Utd", "Red Devils", "United", "Reds"],
    "Middlesbrough": ["Middlesbrough", "Boro"],
    "Newcastle": ["Newcastle United", "Newcastle Utd", "Newcastle", "Magpies", "Toon", "United"],
    "Norwich": ["Norwich City", "Norwich", "Canaries", "City"],
    "QPR": ["Queens Park Rangers", "QPR", "Hoops", "Rangers"],
    "Southampton": ["Southampton", "Saints"],
    "Stoke": ["Stoke City", "Stoke", "Potters", "City"],
    "Sunderland": ["Sunderland", "Black Cats"],
    "Swansea": ["Swansea City", "Swansea", "Swans", "City"],
    "Tottenham": ["Tottenham Hotspur", "Tottenham Hostpur", "Tottenham", "Spurs", "Lilywhites"],
    "Watford": ["Watford", "Hornets"],
    "West Brom": ["West BromwichAlbion",  # typo in one Guardian preview
                  "West Bromwich Albion", "West Bromwich", "West Brom", "Baggies", "Throstles", "Albion"],
    "West Ham": ["West Ham United", "West Ham", "Hammers", "Irons", "United"],
}

# Nicknames that point to more than one club. They are only masked as HOME/AWAY when
# exactly one of the two teams uses them; otherwise they are left (Level 2-3) or
# masked as an unidentified club (Level 4+).
SHARED_NICKNAMES = {"City", "United", "Blues", "Reds", "Albion", "Town", "Rangers"}

# Other clubs that previews mention (cup opponents, European clubs, former clubs).
OTHER_CLUBS = [
    # English clubs outside the 2013-18 Premier League sample
    "Wolverhampton Wanderers", "Wolves", "Leeds United", "Leeds Utd", "Leeds", "Nottingham Forest", "Forest",
    "Derby County", "Derby", "Sheffield United", "Sheffield Wednesday", "Sheff Wed", "Blackburn Rovers",
    "Blackburn", "Bolton Wanderers", "Bolton", "Wigan Athletic", "Wigan", "Reading", "Birmingham City",
    "Ipswich Town", "Ipswich", "Brentford", "Millwall", "Charlton Athletic", "Charlton", "Portsmouth",
    "Leyton Orient", "Accrington Stanley", "MK Dons", "Milton Keynes Dons", "Bradford City", "Bristol City",
    "Bristol Rovers", "Preston North End", "Preston", "Barnsley", "Rotherham", "Blackpool", "Coventry City",
    "Coventry", "Oxford United", "Peterborough United", "Peterborough", "Rochdale", "Shrewsbury Town",
    "Shrewsbury", "Sutton United", "Sutton", "Lincoln City", "Colchester United", "Colchester", "Walsall",
    "Chesterfield", "Bury", "Carlisle United", "Carlisle", "Doncaster Rovers", "Luton Town", "Luton",
    "Burton Albion", "Fleetwood", "Yeovil", "Wycombe", "Notts County", "Scunthorpe", "Southend",
    "Plymouth", "Exeter", "Cambridge United", "Barnet", "Stevenage", "Gillingham", "Brighton & Hove",
    # Scottish and European clubs
    "Celtic", "Glasgow Rangers", "Hearts", "Aberdeen", "Real Madrid", "Barcelona", "Barça", "Atlético Madrid",
    "Atlético", "Atletico Madrid", "Sevilla", "Valencia", "Villarreal", "Celta Vigo", "Athletic Bilbao",
    "Real Sociedad", "Málaga", "Bayern Munich", "Bayern", "Borussia Dortmund", "Dortmund", "Schalke",
    "Bayer Leverkusen", "Leverkusen", "Wolfsburg", "RB Leipzig", "Juventus", "Internazionale", "Inter",
    "Inter Milan", "Milan", "AC Milan", "Roma", "Lazio", "Napoli", "Fiorentina", "Paris Saint-Germain", "PSG",
    "Monaco", "Lyon", "Marseille", "Ajax", "PSV Eindhoven", "PSV", "Feyenoord", "Porto", "Benfica",
    "Sporting Lisbon", "Sporting", "Anderlecht", "Club Brugge", "Basel", "Olympiakos", "Olympiacos",
    "Besiktas", "Galatasaray", "Fenerbahce", "CSKA Moscow", "Zenit St Petersburg", "Zenit", "Spartak Moscow",
    "Dynamo Kyiv", "Dinamo Kyiv", "Dnipro", "Shakhtar Donetsk", "Maribor", "Qarabag", "Dinamo Zagreb",
    "Maccabi Tel Aviv", "Partizan Belgrade", "Red Star Belgrade", "Steaua Bucharest", "Rostov", "Ludogorets",
    "Legia Warsaw", "Apoel", "Astana", "Sion", "Krasnodar", "Club América", "LA Galaxy", "New York City",
    "Orlando City", "Guangzhou Evergrande", "Al-Ahli", "Al-Sadd", "DC United", "D.C. United", "Toronto FC",
    "New York Red Bulls", "Montreal Impact", "Borussia Mönchengladbach", "Mönchengladbach", "Hoffenheim",
    "Eintracht Frankfurt", "Werder Bremen", "Hamburg", "Stuttgart", "Udinese", "Sampdoria", "Torino", "Genoa",
    "Real Zaragoza", "Astra Giurgiu", "Lokomotiv Moscow", "Metz", "Östersund", "Juve", "Bordeaux", "Lille", "Nice", "Rennes", "Twente", "Utrecht", "Uniteds", "Citys",
]

# Stadiums (current and former names) of clubs in or around the sample, plus neutral venues.
VENUES = [
    "Emirates Stadium", "Emirates", "Highbury", "Villa Park", "Vitality Stadium", "Dean Court", "Amex Stadium",
    "Amex", "Falmer Stadium", "Turf Moor", "Cardiff City Stadium", "Stamford Bridge", "Selhurst Park",
    "Goodison Park", "Goodison", "Craven Cottage", "John Smith’s Stadium", "John Smith's Stadium",
    "Kirklees Stadium", "KC Stadium", "KCOM Stadium", "Kcom Stadium", "King Power Stadium", "King Power",
    "Filbert Street", "Anfield", "Etihad Stadium", "Etihad", "City of Manchester Stadium", "Maine Road",
    "Old Trafford", "Riverside Stadium", "Riverside", "St James’ Park", "St James' Park", "St James’s Park",
    "Carrow Road", "Loftus Road", "St Mary’s Stadium", "St Mary’s", "St Mary's", "Britannia Stadium",
    "Britannia", "bet365 Stadium", "Bet365 Stadium", "Stadium of Light", "Liberty Stadium", "Liberty",
    "White Hart Lane", "Wembley Stadium", "Wembley", "Vicarage Road", "The Hawthorns", "Hawthorns",
    "Boleyn Ground", "Upton Park", "London Stadium", "Olympic Stadium", "Molineux", "Elland Road",
    "City Ground", "Pride Park", "Bramall Lane", "Hillsborough", "Ewood Park", "Macron Stadium",
    "Reebok Stadium", "DW Stadium", "Madejski Stadium", "Madejski", "St Andrew’s", "St Andrew's",
    "Portman Road", "Griffin Park", "The Den", "The Valley", "Fratton Park", "Camp Nou", "Bernabéu",
    "Santiago Bernabéu", "Allianz Arena", "Westfalenstadion", "Signal Iduna Park", "San Siro",
    "Parc des Princes", "Wanda Metropolitano", "Metropolitano", "Vicente Calderón", "Juventus Stadium",
    "Stade Vélodrome", "Vélodrome", "Estádio da Luz", "Estádio do Dragão", "Stadio Olimpico", "Mestalla",
    "Stanley Park", "Millennium Stadium", "Principality Stadium",
    # Stands and supporter groups named after them identify the ground
    "Spion Kop", "Kop", "Stretford End", "Gwladys Street", "Holte End", "Shed End", "Gallowgate End",
    "Gallowgate", "Clock End", "North Bank", "Holmesdale Road", "Holmesdale", "Bobby Moore Stand",
]

# Regions, cities and areas used to identify clubs ("the Tyne-Wear derby", "north London").
PLACES = [
    "Tyneside", "Wearside", "Teesside", "Merseyside", "Humberside", "Potteries", "Black Country",
    "West Midlands", "East Midlands", "Midlands", "south coast", "south-coast", "north-east", "north east",
    "North East", "north-west", "north west", "North West", "north London", "south London", "west London",
    "east London", "London", "Manchester", "Birmingham", "Sheffield", "Bristol", "Nottingham", "Yorkshire",
    "Lancashire", "Teeside", "Tyne-Wear", "Tyne", "Wear", "Mersey", "Wales", "south Wales", "Scotland",
    "Lancashire", "Norfolk", "Suffolk", "Hampshire", "Dorset", "Hertfordshire", "Staffordshire", "M23",
    "Wearsiders", "Merseysiders", "Mancunian", "Mancunians", "Geordie", "Geordies", "Scousers", "Evertonians",
    "Cockney", "west-London", "north-London", "south-London", "east-London", "North London", "South London",
    "West London", "East London", "South Wales", "South Coast", "North-East", "North-West", "England",
    "Republic of Ireland", "Northern Ireland", "Ireland",
]

# Nationality nouns ("the Scot", "the Dutchman") identify managers and players.
NATIONALITY_NOUNS = [
    "Scot", "Scotsman", "Dutchman", "Frenchman", "Spaniard", "Englishman", "Welshman", "Irishman", "Northern Irishman",
    "Argentinian", "Argentine", "Italian", "German", "Portuguese", "Belgian", "Brazilian", "Croat", "Croatian", "Serb",
    "Serbian", "Swiss", "Dane", "Swede", "Norwegian", "Uruguayan", "Chilean", "Ivorian", "Senegalese", "Ghanaian",
    "Nigerian", "Algerian", "Austrian", "Pole", "Czech", "Slovak", "Slovenian", "Greek", "Turk", "Moroccan",
    "Egyptian", "Colombian", "Mexican", "American", "Australian", "Japanese", "Korean", "Venezuelan", "Ecuadorian",
    "Costa Rican", "Icelander", "Catalan", "Basque", "Cameroonian", "Malian", "Gabonese", "Togolese", "Congolese",
    "Paraguayan", "Peruvian", "Bosnian", "Montenegrin", "Albanian", "Ukrainian", "Russian", "Israeli", "Iranian",
    "Jamaican", "Canadian", "Kiwi", "Finn", "Hungarian", "Romanian", "Bulgarian", "Spanish", "French", "Dutch",
    "English", "Welsh", "Irish", "Scottish", "British", "Brit",
]

# Never masked: competitions and general terms that carry legitimate stakes information.
KEEP = {
    "Premier League", "Champions League", "Europa League", "FA Cup", "League Cup", "Championship",
    "League One", "League Two", "Football League", "Uefa", "UEFA", "Fifa", "FIFA", "FA", "Europe",
    "European", "VAR", "Golden Boot", "Football Association", "Premier League’s", "Champions League’s",
    "Bundesliga", "La Liga", "Serie A", "Ligue 1", "Twitter", "Instagram", "Facebook", "PFA", "FAW",
    "Fair Play League", "Division One", "Division Two", "Division Three", "Time", "Blue", "No9", "No10", "No2",
    "XI", "No1", "No7", "No8", "No11", "Gracenote", "Opta", "Sky", "BBC", "Match of the Day",
}

# --- Level 5: time and season cues -------------------------------------------------
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "June", "July", "August", "September", "October",
          "November", "December"]  # "May" handled separately (also a verb)
HOLIDAYS = ["Boxing Day", "New Year’s Day", "New Year's Day", "New Year’s Eve", "New Year's Eve", "New Year",
            "new year", "Christmas", "Easter", "Halloween", "Bonfire Night"]
SEASON_PHRASES = [  # matched case-insensitively
    "opening day of the season", "opening day", "opening weekend", "opening game of the season",
    "first game of the season", "first match of the season", "first home game of the season",
    "season opener", "season-opener", "final day of the season", "last day of the season", "final day",
    "final weekend", "final game of the season", "last game of the season", "last match of the season",
    "final match of the season", "final home game of the season", "last home game of the season",
    "international break", "turn of the year", "festive period", "festive fixtures",
]
# Sponsored names reveal the season (Capital One 2012-16, EFL 2016-17, Carabao 2017-).
SPONSORED_CUPS = ["Capital One Cup", "EFL Cup", "Carabao Cup", "Checkatrade Trophy"]
# Tournaments that date a preview (World Cups 2014/2018, Euro 2016 ...).
DATED_EVENTS = ["Community Shield", "World Cup", "Euro", "Euros", "Africa Cup of Nations",
                "Confederations Cup", "Copa América", "Club World Cup", "Olympics"]

# People spaCy mislabels as places or organisations in these previews; always masked as people.
MANUAL_PERSONS = [
    "Diego Costa", "Costa", "N’Golo Kanté", "Kanté", "Roque Mesa", "Chris Wood", "Paulinho", "Rio Ferdinand",
    "Nacho Monreal", "Memphis Depay", "Gökhan Inler", "Gokhan Inler", "Inler", "Cheik Tioté", "Gordon Banks",
    "Lucas Pérez", "Emerson", "Juninho", "César Azpilicueta", "Jesé Rodríguez", "Adrián", "Wilfred Ndidi",
    "Chico", "Neymar", "Zlatan", "Fernandinho", "Dame N’Doye", "Salomón Rondón", "Rondón",
]
