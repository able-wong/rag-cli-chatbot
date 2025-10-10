"""
City to timezone conversion tool.

NOTE: This tool uses a hardcoded city-to-timezone database for demonstration purposes
to illustrate how OpenWebUI-compatible tools can be integrated with the RAG CLI chatbot.
The database includes 500+ cities and aliases but is NOT meant to be comprehensive or
production-ready. For production use, consider using a proper geocoding API like:
- Google Maps Geocoding API
- OpenStreetMap Nominatim
- GeoNames API
- timezonefinder library with lat/long lookup
"""

from pydantic import BaseModel


class Tools:
    """City timezone lookup tools."""

    class Valves(BaseModel):
        """Tool-level configuration (none needed for this tool)."""

        pass

    class UserValves(BaseModel):
        """User-specific configuration (none needed for this tool)."""

        pass

    def __init__(self):
        """Initialize tool with city-to-timezone mapping."""
        self.valves = self.Valves()
        self.user_valves = self.UserValves()

        # Comprehensive city-to-timezone mapping
        self.city_timezones = {
            # ============================================================
            # ASIA - Major cities + Capitals
            # ============================================================
            # East Asia
            "hong kong": "Asia/Hong_Kong",
            "hongkong": "Asia/Hong_Kong",
            "hk": "Asia/Hong_Kong",
            "hong kong china": "Asia/Hong_Kong",
            "hong kong sar": "Asia/Hong_Kong",
            "beijing": "Asia/Shanghai",
            "beijing china": "Asia/Shanghai",
            "shanghai": "Asia/Shanghai",
            "shanghai china": "Asia/Shanghai",
            "guangzhou": "Asia/Shanghai",
            "shenzhen": "Asia/Shanghai",
            "chengdu": "Asia/Shanghai",
            "chongqing": "Asia/Shanghai",
            "hangzhou": "Asia/Shanghai",
            "wuhan": "Asia/Shanghai",
            "xi'an": "Asia/Shanghai",
            "xian": "Asia/Shanghai",
            "nanjing": "Asia/Shanghai",
            "tianjin": "Asia/Shanghai",
            "suzhou": "Asia/Shanghai",
            "harbin": "Asia/Shanghai",
            "qingdao": "Asia/Shanghai",
            "dalian": "Asia/Shanghai",
            "jinan": "Asia/Shanghai",
            "taipei": "Asia/Taipei",
            "taipei taiwan": "Asia/Taipei",
            "tokyo": "Asia/Tokyo",
            "tokyo japan": "Asia/Tokyo",
            "osaka": "Asia/Tokyo",
            "kyoto": "Asia/Tokyo",
            "yokohama": "Asia/Tokyo",
            "nagoya": "Asia/Tokyo",
            "sapporo": "Asia/Tokyo",
            "fukuoka": "Asia/Tokyo",
            "seoul": "Asia/Seoul",
            "seoul south korea": "Asia/Seoul",
            "seoul korea": "Asia/Seoul",
            "busan": "Asia/Seoul",
            "incheon": "Asia/Seoul",
            "pyongyang": "Asia/Pyongyang",
            "ulaanbaatar": "Asia/Ulaanbaatar",
            # Southeast Asia
            "singapore": "Asia/Singapore",
            "singapore sg": "Asia/Singapore",
            "bangkok": "Asia/Bangkok",
            "bangkok thailand": "Asia/Bangkok",
            "kuala lumpur": "Asia/Kuala_Lumpur",
            "kuala lumpur malaysia": "Asia/Kuala_Lumpur",
            "jakarta": "Asia/Jakarta",
            "jakarta indonesia": "Asia/Jakarta",
            "manila": "Asia/Manila",
            "manila philippines": "Asia/Manila",
            "ho chi minh": "Asia/Ho_Chi_Minh",
            "hanoi": "Asia/Bangkok",
            "phnom penh": "Asia/Phnom_Penh",
            "vientiane": "Asia/Vientiane",
            "yangon": "Asia/Yangon",
            "naypyidaw": "Asia/Yangon",
            "bandar seri begawan": "Asia/Brunei",
            "dili": "Asia/Dili",
            # South Asia
            "mumbai": "Asia/Kolkata",
            "delhi": "Asia/Kolkata",
            "new delhi": "Asia/Kolkata",
            "bangalore": "Asia/Kolkata",
            "kolkata": "Asia/Kolkata",
            "chennai": "Asia/Kolkata",
            "hyderabad": "Asia/Kolkata",
            "pune": "Asia/Kolkata",
            "ahmedabad": "Asia/Kolkata",
            "karachi": "Asia/Karachi",
            "islamabad": "Asia/Karachi",
            "lahore": "Asia/Karachi",
            "dhaka": "Asia/Dhaka",
            "colombo": "Asia/Colombo",
            "kathmandu": "Asia/Kathmandu",
            "thimphu": "Asia/Thimphu",
            "male": "Indian/Maldives",
            # Central Asia
            "tashkent": "Asia/Tashkent",
            "almaty": "Asia/Almaty",
            "astana": "Asia/Almaty",
            "bishkek": "Asia/Bishkek",
            "dushanbe": "Asia/Dushanbe",
            "ashgabat": "Asia/Ashgabat",
            # Middle East
            "dubai": "Asia/Dubai",
            "abu dhabi": "Asia/Dubai",
            "riyadh": "Asia/Riyadh",
            "jeddah": "Asia/Riyadh",
            "mecca": "Asia/Riyadh",
            "doha": "Asia/Qatar",
            "kuwait city": "Asia/Kuwait",
            "manama": "Asia/Bahrain",
            "muscat": "Asia/Muscat",
            "jerusalem": "Asia/Jerusalem",
            "tel aviv": "Asia/Jerusalem",
            "amman": "Asia/Amman",
            "beirut": "Asia/Beirut",
            "damascus": "Asia/Damascus",
            "baghdad": "Asia/Baghdad",
            "tehran": "Asia/Tehran",
            "kabul": "Asia/Kabul",
            "ankara": "Europe/Istanbul",
            "istanbul": "Europe/Istanbul",
            "tbilisi": "Asia/Tbilisi",
            "yerevan": "Asia/Yerevan",
            "baku": "Asia/Baku",
            "nicosia": "Asia/Nicosia",

            # ============================================================
            # NORTH AMERICA - US Cities (100+)
            # ============================================================
            # US - East Coast
            "new york": "America/New_York",
            "new york ny": "America/New_York",
            "new york city": "America/New_York",
            "new york usa": "America/New_York",
            "nyc": "America/New_York",
            "brooklyn": "America/New_York",
            "queens": "America/New_York",
            "manhattan": "America/New_York",
            "boston": "America/New_York",
            "boston ma": "America/New_York",
            "boston massachusetts": "America/New_York",
            "boston usa": "America/New_York",
            "philadelphia": "America/New_York",
            "philadelphia pa": "America/New_York",
            "baltimore": "America/New_York",
            "washington": "America/New_York",
            "dc": "America/New_York",
            "washington dc": "America/New_York",
            "miami": "America/New_York",
            "miami fl": "America/New_York",
            "miami florida": "America/New_York",
            "atlanta": "America/New_York",
            "atlanta ga": "America/New_York",
            "atlanta georgia": "America/New_York",
            "charlotte": "America/New_York",
            "orlando": "America/New_York",
            "tampa": "America/New_York",
            "jacksonville": "America/New_York",
            "richmond": "America/New_York",
            "raleigh": "America/New_York",
            "pittsburgh": "America/New_York",
            "buffalo": "America/New_York",
            "rochester": "America/New_York",
            "syracuse": "America/New_York",
            "albany": "America/New_York",
            "newark": "America/New_York",
            "jersey city": "America/New_York",
            "providence": "America/New_York",
            "hartford": "America/New_York",
            "portland me": "America/New_York",  # Portland, Maine
            "savannah": "America/New_York",
            "charleston": "America/New_York",
            "columbus": "America/New_York",
            "cleveland": "America/New_York",
            "cincinnati": "America/New_York",
            "detroit": "America/Detroit",
            "indianapolis": "America/Indiana/Indianapolis",
            "louisville": "America/Kentucky/Louisville",
            "nashville": "America/Chicago",
            "memphis": "America/Chicago",
            "knoxville": "America/New_York",
            # US - Central
            "chicago": "America/Chicago",
            "chicago il": "America/Chicago",
            "chicago illinois": "America/Chicago",
            "chicago usa": "America/Chicago",
            "dallas": "America/Chicago",
            "dallas tx": "America/Chicago",
            "dallas texas": "America/Chicago",
            "houston": "America/Chicago",
            "houston tx": "America/Chicago",
            "houston texas": "America/Chicago",
            "austin": "America/Chicago",
            "san antonio": "America/Chicago",
            "fort worth": "America/Chicago",
            "el paso": "America/Denver",
            "oklahoma city": "America/Chicago",
            "tulsa": "America/Chicago",
            "kansas city": "America/Chicago",
            "st louis": "America/Chicago",
            "minneapolis": "America/Chicago",
            "milwaukee": "America/Chicago",
            "madison": "America/Chicago",
            "des moines": "America/Chicago",
            "omaha": "America/Chicago",
            "new orleans": "America/Chicago",
            "baton rouge": "America/Chicago",
            "birmingham al": "America/Chicago",  # Birmingham, Alabama
            "little rock": "America/Chicago",
            "jackson": "America/Chicago",
            "wichita": "America/Chicago",
            "lincoln": "America/Chicago",
            "fargo": "America/Chicago",
            "sioux falls": "America/Chicago",
            "cedar rapids": "America/Chicago",
            "springfield": "America/Chicago",
            # US - Mountain
            "denver": "America/Denver",
            "colorado springs": "America/Denver",
            "boulder": "America/Denver",
            "salt lake city": "America/Denver",
            "albuquerque": "America/Denver",
            "santa fe": "America/Denver",
            "phoenix": "America/Phoenix",
            "tucson": "America/Phoenix",
            "scottsdale": "America/Phoenix",
            "mesa": "America/Phoenix",
            "boise": "America/Boise",
            "billings": "America/Denver",
            "casper": "America/Denver",
            "cheyenne": "America/Denver",
            # US - West Coast
            "los angeles": "America/Los_Angeles",
            "los angeles ca": "America/Los_Angeles",
            "los angeles california": "America/Los_Angeles",
            "la": "America/Los_Angeles",
            "san diego": "America/Los_Angeles",
            "san diego ca": "America/Los_Angeles",
            "san diego california": "America/Los_Angeles",
            "san francisco": "America/Los_Angeles",
            "san francisco ca": "America/Los_Angeles",
            "san francisco california": "America/Los_Angeles",
            "sf": "America/Los_Angeles",
            "san jose": "America/Los_Angeles",
            "san jose ca": "America/Los_Angeles",
            "san jose california": "America/Los_Angeles",
            "oakland": "America/Los_Angeles",
            "sacramento": "America/Los_Angeles",
            "fresno": "America/Los_Angeles",
            "bakersfield": "America/Los_Angeles",
            "long beach": "America/Los_Angeles",
            "anaheim": "America/Los_Angeles",
            "santa barbara": "America/Los_Angeles",
            "riverside": "America/Los_Angeles",
            "irvine": "America/Los_Angeles",
            "seattle": "America/Los_Angeles",
            "seattle wa": "America/Los_Angeles",
            "seattle washington": "America/Los_Angeles",
            "seattle usa": "America/Los_Angeles",
            "portland": "America/Los_Angeles",  # Portland, Oregon (default)
            "portland or": "America/Los_Angeles",  # Portland, Oregon
            "portland oregon": "America/Los_Angeles",
            "spokane": "America/Los_Angeles",
            "tacoma": "America/Los_Angeles",
            "eugene": "America/Los_Angeles",
            "salem": "America/Los_Angeles",
            "las vegas": "America/Los_Angeles",
            "reno": "America/Los_Angeles",
            # US - Alaska & Hawaii
            "anchorage": "America/Anchorage",
            "juneau": "America/Juneau",
            "fairbanks": "America/Anchorage",
            "honolulu": "Pacific/Honolulu",
            # Canada - Major Cities
            "toronto": "America/Toronto",
            "toronto on": "America/Toronto",
            "toronto ontario": "America/Toronto",
            "toronto canada": "America/Toronto",
            "ottawa": "America/Toronto",
            "ottawa on": "America/Toronto",
            "ottawa ontario": "America/Toronto",
            "ottawa canada": "America/Toronto",
            "montreal": "America/Toronto",
            "montreal qc": "America/Toronto",
            "montreal quebec": "America/Toronto",
            "montreal canada": "America/Toronto",
            "vancouver": "America/Vancouver",
            "vancouver bc": "America/Vancouver",
            "vancouver british columbia": "America/Vancouver",
            "vancouver canada": "America/Vancouver",
            "calgary": "America/Edmonton",
            "calgary ab": "America/Edmonton",
            "calgary alberta": "America/Edmonton",
            "calgary canada": "America/Edmonton",
            "edmonton": "America/Edmonton",
            "edmonton ab": "America/Edmonton",
            "edmonton alberta": "America/Edmonton",
            "edmonton canada": "America/Edmonton",
            "winnipeg": "America/Winnipeg",
            "winnipeg mb": "America/Winnipeg",
            "winnipeg manitoba": "America/Winnipeg",
            "winnipeg canada": "America/Winnipeg",
            "quebec city": "America/Toronto",
            "hamilton": "America/Toronto",
            "kitchener": "America/Toronto",
            "london on": "America/Toronto",  # London, Ontario
            "london ontario": "America/Toronto",
            "victoria bc": "America/Vancouver",  # Victoria, BC
            "halifax": "America/Halifax",
            "saskatoon": "America/Regina",
            "regina": "America/Regina",
            "st johns": "America/St_Johns",
            "fredericton": "America/Moncton",
            "charlottetown": "America/Halifax",
            "whitehorse": "America/Whitehorse",
            "yellowknife": "America/Yellowknife",
            "iqaluit": "America/Iqaluit",
            # Mexico & Central America
            "mexico city": "America/Mexico_City",
            "guadalajara": "America/Mexico_City",
            "monterrey": "America/Monterrey",
            "cancun": "America/Cancun",
            "tijuana": "America/Tijuana",
            "ciudad juarez": "America/Ojinaga",
            "guatemala city": "America/Guatemala",
            "san salvador": "America/El_Salvador",
            "tegucigalpa": "America/Tegucigalpa",
            "managua": "America/Managua",
            "san jose cr": "America/Costa_Rica",  # San Jose, Costa Rica
            "san jose costa rica": "America/Costa_Rica",
            "panama city": "America/Panama",
            "havana": "America/Havana",
            "kingston": "America/Jamaica",
            "port au prince": "America/Port-au-Prince",
            "santo domingo": "America/Santo_Domingo",
            "san juan": "America/Puerto_Rico",

            # ============================================================
            # EUROPE - Major Cities (80+)
            # ============================================================
            # Western Europe
            "london": "Europe/London",  # London, UK (default)
            "london uk": "Europe/London",
            "london england": "Europe/London",
            "london united kingdom": "Europe/London",
            "birmingham": "Europe/London",  # Birmingham, UK (default)
            "birmingham uk": "Europe/London",
            "birmingham england": "Europe/London",
            "manchester": "Europe/London",
            "manchester uk": "Europe/London",
            "manchester england": "Europe/London",
            "liverpool": "Europe/London",
            "leeds": "Europe/London",
            "glasgow": "Europe/London",
            "edinburgh": "Europe/London",
            "cardiff": "Europe/London",
            "belfast": "Europe/Belfast",
            "paris": "Europe/Paris",
            "paris france": "Europe/Paris",
            "marseille": "Europe/Paris",
            "lyon": "Europe/Paris",
            "toulouse": "Europe/Paris",
            "nice": "Europe/Paris",
            "nantes": "Europe/Paris",
            "strasbourg": "Europe/Paris",
            "bordeaux": "Europe/Paris",
            "lille": "Europe/Paris",
            "brussels": "Europe/Brussels",
            "antwerp": "Europe/Brussels",
            "ghent": "Europe/Brussels",
            "amsterdam": "Europe/Amsterdam",
            "amsterdam netherlands": "Europe/Amsterdam",
            "rotterdam": "Europe/Amsterdam",
            "the hague": "Europe/Amsterdam",
            "utrecht": "Europe/Amsterdam",
            "luxembourg city": "Europe/Luxembourg",
            "monaco": "Europe/Monaco",
            # Germanic Europe
            "berlin": "Europe/Berlin",
            "berlin germany": "Europe/Berlin",
            "munich": "Europe/Berlin",
            "frankfurt": "Europe/Berlin",
            "hamburg": "Europe/Berlin",
            "cologne": "Europe/Berlin",
            "stuttgart": "Europe/Berlin",
            "dusseldorf": "Europe/Berlin",
            "dortmund": "Europe/Berlin",
            "dresden": "Europe/Berlin",
            "leipzig": "Europe/Berlin",
            "vienna": "Europe/Vienna",
            "vienna austria": "Europe/Vienna",
            "graz": "Europe/Vienna",
            "salzburg": "Europe/Vienna",
            "zurich": "Europe/Zurich",
            "zurich switzerland": "Europe/Zurich",
            "geneva": "Europe/Zurich",
            "basel": "Europe/Zurich",
            "bern": "Europe/Zurich",
            "lausanne": "Europe/Zurich",
            "liechtenstein": "Europe/Vaduz",
            "vaduz": "Europe/Vaduz",
            # Southern Europe
            "madrid": "Europe/Madrid",
            "madrid spain": "Europe/Madrid",
            "barcelona": "Europe/Madrid",
            "barcelona spain": "Europe/Madrid",
            "valencia": "Europe/Madrid",
            "seville": "Europe/Madrid",
            "bilbao": "Europe/Madrid",
            "malaga": "Europe/Madrid",
            "lisbon": "Europe/Lisbon",
            "porto": "Europe/Lisbon",
            "rome": "Europe/Rome",
            "rome italy": "Europe/Rome",
            "milan": "Europe/Rome",
            "milan italy": "Europe/Rome",
            "naples": "Europe/Rome",
            "turin": "Europe/Rome",
            "florence": "Europe/Rome",
            "venice": "Europe/Rome",
            "bologna": "Europe/Rome",
            "vatican city": "Europe/Vatican",
            "athens": "Europe/Athens",
            "thessaloniki": "Europe/Athens",
            "valletta": "Europe/Malta",
            "nicosia cy": "Asia/Nicosia",  # Nicosia, Cyprus (Europe section)
            # Eastern Europe
            "moscow": "Europe/Moscow",
            "moscow russia": "Europe/Moscow",
            "saint petersburg": "Europe/Moscow",
            "st petersburg": "Europe/Moscow",
            "novosibirsk": "Asia/Novosibirsk",
            "yekaterinburg": "Asia/Yekaterinburg",
            "kyiv": "Europe/Kyiv",
            "kiev": "Europe/Kyiv",
            "kharkiv": "Europe/Kyiv",
            "odesa": "Europe/Kyiv",
            "minsk": "Europe/Minsk",
            "warsaw": "Europe/Warsaw",
            "krakow": "Europe/Warsaw",
            "gdansk": "Europe/Warsaw",
            "prague": "Europe/Prague",
            "brno": "Europe/Prague",
            "budapest": "Europe/Budapest",
            "bucharest": "Europe/Bucharest",
            "sofia": "Europe/Sofia",
            "belgrade": "Europe/Belgrade",
            "zagreb": "Europe/Zagreb",
            "ljubljana": "Europe/Ljubljana",
            "bratislava": "Europe/Bratislava",
            "vilnius": "Europe/Vilnius",
            "riga": "Europe/Riga",
            "tallinn": "Europe/Tallinn",
            "helsinki": "Europe/Helsinki",
            "chisinau": "Europe/Chisinau",
            "tirana": "Europe/Tirane",
            "skopje": "Europe/Skopje",
            "pristina": "Europe/Belgrade",
            "sarajevo": "Europe/Sarajevo",
            "podgorica": "Europe/Podgorica",
            # Nordic Countries
            "stockholm": "Europe/Stockholm",
            "gothenburg": "Europe/Stockholm",
            "malmo": "Europe/Stockholm",
            "oslo": "Europe/Oslo",
            "bergen": "Europe/Oslo",
            "copenhagen": "Europe/Copenhagen",
            "aarhus": "Europe/Copenhagen",
            "reykjavik": "Atlantic/Reykjavik",
            "dublin": "Europe/Dublin",
            "cork": "Europe/Dublin",
            # Turkey (already defined in Asia section, removing duplicates)

            # ============================================================
            # OCEANIA
            # ============================================================
            "sydney": "Australia/Sydney",
            "sydney australia": "Australia/Sydney",
            "sydney nsw": "Australia/Sydney",
            "melbourne": "Australia/Melbourne",
            "melbourne australia": "Australia/Melbourne",
            "melbourne vic": "Australia/Melbourne",
            "brisbane": "Australia/Brisbane",
            "perth": "Australia/Perth",
            "adelaide": "Australia/Adelaide",
            "gold coast": "Australia/Brisbane",
            "canberra": "Australia/Sydney",
            "hobart": "Australia/Hobart",
            "darwin": "Australia/Darwin",
            "auckland": "Pacific/Auckland",
            "wellington": "Pacific/Auckland",
            "christchurch": "Pacific/Auckland",
            "suva": "Pacific/Fiji",
            "port moresby": "Pacific/Port_Moresby",
            "noumea": "Pacific/Noumea",
            "papeete": "Pacific/Tahiti",
            "apia": "Pacific/Apia",
            "nuku'alofa": "Pacific/Tongatapu",
            "port vila": "Pacific/Efate",
            "honiara": "Pacific/Guadalcanal",

            # ============================================================
            # SOUTH AMERICA
            # ============================================================
            "sao paulo": "America/Sao_Paulo",
            "rio de janeiro": "America/Sao_Paulo",
            "brasilia": "America/Sao_Paulo",
            "salvador": "America/Sao_Paulo",
            "fortaleza": "America/Fortaleza",
            "belo horizonte": "America/Sao_Paulo",
            "manaus": "America/Manaus",
            "curitiba": "America/Sao_Paulo",
            "recife": "America/Recife",
            "porto alegre": "America/Sao_Paulo",
            "buenos aires": "America/Argentina/Buenos_Aires",
            "cordoba": "America/Argentina/Cordoba",
            "rosario": "America/Argentina/Cordoba",
            "mendoza": "America/Argentina/Mendoza",
            "santiago": "America/Santiago",
            "valparaiso": "America/Santiago",
            "lima": "America/Lima",
            "arequipa": "America/Lima",
            "bogota": "America/Bogota",
            "medellin": "America/Bogota",
            "cali": "America/Bogota",
            "caracas": "America/Caracas",
            "quito": "America/Guayaquil",
            "guayaquil": "America/Guayaquil",
            "la paz": "America/La_Paz",
            "santa cruz": "America/La_Paz",
            "asuncion": "America/Asuncion",
            "montevideo": "America/Montevideo",
            "paramaribo": "America/Paramaribo",
            "cayenne": "America/Cayenne",
            "georgetown": "America/Guyana",

            # ============================================================
            # AFRICA
            # ============================================================
            "cairo": "Africa/Cairo",
            "alexandria": "Africa/Cairo",
            "johannesburg": "Africa/Johannesburg",
            "cape town": "Africa/Johannesburg",
            "durban": "Africa/Johannesburg",
            "pretoria": "Africa/Johannesburg",
            "lagos": "Africa/Lagos",
            "abuja": "Africa/Lagos",
            "nairobi": "Africa/Nairobi",
            "mombasa": "Africa/Nairobi",
            "casablanca": "Africa/Casablanca",
            "rabat": "Africa/Casablanca",
            "marrakech": "Africa/Casablanca",
            "algiers": "Africa/Algiers",
            "tunis": "Africa/Tunis",
            "tripoli": "Africa/Tripoli",
            "khartoum": "Africa/Khartoum",
            "addis ababa": "Africa/Addis_Ababa",
            "dar es salaam": "Africa/Dar_es_Salaam",
            "kampala": "Africa/Kampala",
            "kigali": "Africa/Kigali",
            "lusaka": "Africa/Lusaka",
            "harare": "Africa/Harare",
            "maputo": "Africa/Maputo",
            "antananarivo": "Africa/Nairobi",
            "windhoek": "Africa/Windhoek",
            "gaborone": "Africa/Gaborone",
            "dakar": "Africa/Dakar",
            "bamako": "Africa/Bamako",
            "conakry": "Africa/Conakry",
            "abidjan": "Africa/Abidjan",
            "accra": "Africa/Accra",
            "lome": "Africa/Lome",
            "cotonou": "Africa/Porto-Novo",
            "niamey": "Africa/Niamey",
            "ouagadougou": "Africa/Ouagadougou",
            "monrovia": "Africa/Monrovia",
            "freetown": "Africa/Freetown",
            "luanda": "Africa/Luanda",
            "kinshasa": "Africa/Kinshasa",
            "brazzaville": "Africa/Brazzaville",
            "libreville": "Africa/Libreville",
            "yaounde": "Africa/Douala",
            "bangui": "Africa/Bangui",
            "ndjamena": "Africa/Ndjamena",
            "mogadishu": "Africa/Mogadishu",
            "djibouti": "Africa/Djibouti",
            "asmara": "Africa/Asmara",
            "port louis": "Indian/Mauritius",
            "victoria seychelles": "Indian/Mahe",  # Victoria, Seychelles
        }

    def _normalize_city_name(self, city: str) -> str:
        """
        Normalize city name for consistent lookup.

        Handles:
        - Case conversion (lowercase)
        - Whitespace trimming and normalization
        - Punctuation removal (commas, periods, etc.)
        - Multiple spaces -> single space

        Examples:
            "Toronto, Ontario" -> "toronto ontario"
            "New York,  NY" -> "new york ny"
            "Hong Kong, China" -> "hong kong china"
            "San Francisco, CA." -> "san francisco ca"

        :param city: Raw city name input
        :return: Normalized city name for dictionary lookup
        """
        # Convert to lowercase
        normalized = city.lower()

        # Remove common punctuation (commas, periods, semicolons)
        # Keep hyphens and apostrophes (for cities like "Saint-Petersburg", "Nuku'alofa")
        normalized = normalized.replace(",", " ")
        normalized = normalized.replace(".", " ")
        normalized = normalized.replace(";", " ")

        # Normalize whitespace: strip and collapse multiple spaces to single space
        normalized = " ".join(normalized.split())

        return normalized

    def get_timezone_for_city(self, city: str) -> str:
        """
        Get the timezone for a given city.

        :param city: City name (e.g., 'Hong Kong', 'New York', 'London').
                     For better matching, you can optionally include province/state/country
                     (e.g., 'Toronto Canada', 'Toronto Ontario', 'Hong Kong China',
                     'Birmingham UK', 'San Jose California').
                     Accepts various formats: "Toronto, Ontario", "Toronto, ON", "Toronto Canada", etc.
        :return: The IANA timezone identifier for the city.
        """
        # Normalize city name for lookup
        city_normalized = self._normalize_city_name(city)

        # Look up timezone
        timezone = self.city_timezones.get(city_normalized)

        if timezone:
            return f"The timezone for {city} is: {timezone}"
        else:
            # Provide helpful suggestions
            # Find similar cities (simple fuzzy matching)
            suggestions = []
            for known_city in self.city_timezones.keys():
                if city_normalized in known_city or known_city in city_normalized:
                    suggestions.append(known_city)

            if suggestions:
                suggestion_str = ", ".join(suggestions[:5])
                return (
                    f"City '{city}' not found in database. "
                    f"Did you mean: {suggestion_str}? "
                    f"Total cities in database: {len(self.city_timezones)}"
                )
            else:
                # Show some example cities
                example_cities = list(self.city_timezones.keys())[:10]
                return (
                    f"City '{city}' not found in database. "
                    f"Example cities: {', '.join(example_cities)}. "
                    f"Total cities in database: {len(self.city_timezones)}"
                )

    def list_supported_cities(self) -> str:
        """
        List all supported cities in the database.

        :return: A formatted list of all supported cities grouped by region.
        """
        # Group cities by region (based on timezone prefix)
        regions = {}
        for city, tz in self.city_timezones.items():
            region = tz.split("/")[0]  # e.g., "Asia", "America", "Europe"
            if region not in regions:
                regions[region] = []
            regions[region].append(city)

        # Format output
        output_lines = [f"Supported cities ({len(self.city_timezones)} total):\n"]

        for region in sorted(regions.keys()):
            cities = sorted(regions[region])
            output_lines.append(f"\n{region} ({len(cities)} cities):")
            # Show first 10 cities per region
            cities_to_show = cities[:10]
            output_lines.append(f"  {', '.join(cities_to_show)}")
            if len(cities) > 10:
                output_lines.append(f"  ... and {len(cities) - 10} more")

        return "\n".join(output_lines)
