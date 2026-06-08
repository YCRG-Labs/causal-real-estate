"""Prompt templates for counterfactual listing rewrites.

Two templates: style_swap and style_stripped. Both follow the prompt-engineering
recipe from the dossier (Madaan et al. 2021; Dixit et al. CORE 2022; Bhattacharjee
& Liu 2024):

  - explicit "preserve X" / "change Y" constraint lists
  - slot-fact JSON the model must verify against
  - structured JSON output schema (`rewritten_text`, `preserved_slots`)
  - chain-of-thought verification step inlined in the instructions

Output schema is JSON-only so the generator can parse without regex hacks.

Two API surfaces:

  Legacy (uncached):  style_swap_prompt() / style_stripped_prompt()
                      Single-string prompts. Backward compatible.

  Cached (preferred): style_swap_blocks() / style_stripped_blocks()
                      Returns {"system": <constant>, "user": <variable>} so the
                      generator can mark the system prompt with cache_control:
                      ephemeral. Anthropic Sonnet 3.5 caches at 1024 token
                      minimum; the system templates here are sized to clear it.
"""
from __future__ import annotations

import json
from typing import Optional

SUBMARKET_HINTS: dict[str, dict[str, str]] = {
    "sf": {
        "Mission District": (
            "vibrant Latinx culture, mural-lined alleys, taquerias, Dolores Park, "
            "Valencia Street nightlife, victorian and edwardian flats, walkable, "
            "BART-adjacent, working-class roots blending with newer cafes"
        ),
        "Pacific Heights": (
            "elite mansion district, panoramic Bay views, Lyon Street stairs, "
            "Fillmore Street boutiques, manicured garden squares, prestigious, "
            "stately period architecture, blue-chip address"
        ),
        "Sunset": (
            "fog-cooled residential calm, classic two-story stucco homes, family "
            "neighborhood, Ocean Beach, Golden Gate Park access, quiet avenues, "
            "Asian-American community, value-oriented vs. east-side prices"
        ),
        "SoMa": (
            "live/work conversions, tech-corridor energy, modern condo towers, "
            "warehouse loft aesthetic, walkable to Caltrain and Salesforce Park, "
            "boutique restaurants, dynamic urban grit"
        ),
        "Noe Valley": (
            "stroller-friendly Castro-adjacent village feel, sun-pocket microclimate, "
            "24th Street shopping, edwardian single-family homes, family-oriented, "
            "premium school district, quietly affluent"
        ),
        "Castro": (
            "iconic LGBTQ+ neighborhood, rainbow-flag Castro Street, victorian flats, "
            "walkable cafe and nightlife scene, Harvey Milk legacy, vibrant social fabric"
        ),
        "Marina": (
            "waterfront Marina Green and Crissy Field, Mediterranean revival "
            "architecture, Chestnut Street boutiques, athleisure energy, young "
            "professional crowd, sweeping Bay and Golden Gate views"
        ),
        "Richmond": (
            "Pan-Asian culinary corridor along Clement Street, Golden Gate Park "
            "frontage, foggy residential calm, classic SF row houses, established "
            "immigrant communities"
        ),
    },
    "boston": {
        "Back Bay": "Victorian brownstones, Newbury Street boutiques, Copley Square, Boston Public Library, Prudential and Hancock towers, alley parking, Green Line and Orange Line access, walkable to Public Garden, professional and old-money register, brunch-heavy weekends",
        "Beacon Hill": "gas-lit lanterns, cobblestone Acorn Street, federal-style rowhouses, brick sidewalks, Charles Street antique shops, State House gold dome, Boston Common adjacent, Red Line at Park, deeply wealthy, historic district restrictions, low-rise and quiet",
        "North End": "Little Italy, Hanover Street cafes, Mike's Pastry cannoli lines, Old North Church, Paul Revere House, narrow colonial lanes, tight walk-up apartments, Haymarket and Greenway adjacent, Orange Line at North Station, tourist-heavy but tight-knit Italian-American roots",
        "South End": "Victorian bowfront brownstones, Tremont Street restaurants, SoWa Open Market and art galleries, LGBTQ+ heart of Boston, tree-lined parks and squares, Orange Line at Back Bay, design-district lofts, young professional and creative-class register, renovated rowhouses with garden levels",
        "Jamaica Plain": "Arnold Arboretum, Centre Street main drag, Jamaica Pond loop, triple-deckers and Victorians, Orange Line corridor, Latinx Hyde Square pocket, queer-friendly, bike-lane culture, breweries and coffee roasters, progressive politics, working-to-middle class blending with newer professionals",
        "Cambridge": "Harvard Yard, brick sidewalks, indie bookstores, Red Line hub, academic and grad-student register, Charles River paths, colonial and Victorian frame houses mixed with prewar apartments, Cambridge Common, walkable to MIT-bound bus lines, café-and-laptop daytime, international student presence",
        "Brookline": "Coolidge Corner crossroads, Beacon Street Green Line C trolley, leafy side streets, prewar brick apartment buildings, established Jewish community institutions, top public schools, Coolidge Corner Theatre, family-oriented, suburb-feel inside the urban core, stroller-heavy sidewalks",
        "Allston": "student-dense triple-deckers, Harvard Avenue thrift shops, Paradise Rock Club and O'Brien's live music, Allston Christmas curb-furniture, Green Line B trolley, BU and BC commuters, cheap eats and dive bars, diverse immigrant pockets, gritty rock-scene register, high lease turnover",
    },
    "nyc": {
        "Upper East Side": "limestone and prewar coop towers, Museum Mile, Central Park East frontage, Madison Avenue luxury retail, 4/5/6 Lexington line plus Second Avenue Q, doorman buildings, private-school uniforms, old-money and finance register, quiet residential side streets, Carl Schurz Park along the East River",
        "Upper West Side": "prewar elevator buildings, Central Park West and Riverside Drive frontages, Lincoln Center, Zabar's and Fairway food culture, 1/2/3 and B/C lines, stroller-heavy sidewalks, liberal intellectual register, American Museum of Natural History, brownstone side blocks, established Jewish and academic families",
        "Williamsburg": "industrial-to-luxury waterfront, McCarren Park, Bedford Avenue and Berry Street, L train and J/M/Z lines, converted warehouse lofts and new-build glass towers, Domino Park along the East River, Hasidic South Side and hipster North Side split, music venues and rooftop bars, fashion and design industry register, East River Ferry",
        "Bushwick": "Bushwick Collective murals along Jefferson Street, L train at Jefferson and Morgan, former industrial warehouses turned lofts and venues, Latinx working-class roots blending with artist and DIY-show scene, Maria Hernandez Park, raw warehouse aesthetic, queer nightlife, affordable-to-rising rents, taquerias and bodegas",
        "Park Slope": "Prospect Park frontage, brownstone-lined side streets, 5th Avenue and 7th Avenue retail corridors, Park Slope Food Coop, F/G and R subway access, stroller-dense family register, highly rated public schools, Halloween Parade, progressive politics, well-preserved Historic District rowhouses",
        "Astoria": "Greek-American diners and tavernas, Steinway Street commercial spine, N and W trains under the el, Astoria Park along the East River, prewar brick walk-ups and rowhouses, Egyptian and Brazilian and Bangladeshi pockets, Museum of the Moving Image, quieter residential register, outdoor cafes and hookah lounges, more affordable than Manhattan",
        "Harlem": "Apollo Theater on 125th Street, brownstone rows including Striver's Row, Sylvia's and soul-food legacy, jazz history and gospel churches, A/B/C/D and 2/3 lines, Marcus Garvey Park, Black cultural capital with newer mixed-income development, Lenox Avenue and Frederick Douglass Boulevard, prewar elevator buildings",
        "LES": "tenement walk-ups and new luxury glass, Katz's Delicatessen on Houston, Essex Market and Essex Crossing, Tenement Museum, F/J/M/Z at Delancey, immigrant Jewish and Chinese and Puerto Rican layers, Ludlow Street bars and galleries, fire-escape facades, nightlife-heavy weekends, art and music scene",
    },
    "dc": {
        "Georgetown": "federal-style brick rowhouses, cobblestone side streets, M Street and Wisconsin Avenue retail, C&O Canal towpath, Georgetown University, Potomac waterfront, no Metro stop, old-money register, embassies and Foreign Service families, boutique shopping and historic district restrictions",
        "Capitol Hill": "Capitol dome and Library of Congress backdrop, Eastern Market on weekends, Barracks Row along 8th Street SE, rowhouses in pastel and brick, Blue/Orange/Silver Lines, congressional staffer and lobbyist register, Lincoln Park and Stanton Park squares, walkable to Union Station, historic district protections",
        "Dupont Circle": "the fountain and circle itself, embassy row spilling up Massachusetts Avenue, Red Line at Dupont, Connecticut Avenue restaurants and bookstores, Victorian and Beaux-Arts rowhouses converted to condos, long-standing LGBTQ+ presence, Sunday farmers market, young-professional and policy-shop register, walkable nightlife",
        "U Street": "Ben's Chili Bowl on U Street, 9:30 Club nearby on V Street, Black Broadway jazz heritage, Lincoln Theatre, Green and Yellow Lines at U Street, rowhouses and new mid-rise condos, African-American cultural roots with rapid newer-bar influx, 14th Street corridor restaurants, mural-heavy alleys, nightlife-dense",
        "Adams Morgan": "18th Street nightlife strip, jumbo slice late-night pizza, Madam's Organ live blues, eclectic and Latin American immigrant layers, Victorian rowhouses on hilly side streets, no direct Metro, walk to Woodley Park or Columbia Heights, mural art, young-renter and party register on weekends, daytime cafe-and-vintage browsing",
        "NoMa": "North of Massachusetts Avenue new-build mid-rises, Union Market food hall, Red Line at NoMa-Gallaudet U, Metropolitan Branch Trail bike path, warehouse-and-rail industrial bones, Gallaudet University adjacent, young-professional and federal-worker register, ground-floor retail and breweries, Alethia Tanner Park, walk to Union Station",
        "Petworth": "Wardman-style rowhouses with deep front porches, Georgia Avenue and Upshur Street intersection, Green Line at Petworth, Porchfest neighborhood music tradition, walkable food strip on Upshur, long-tenured Black families alongside newer professional buyers, bike-lane improvements, more affordable Northwest tier, family-residential register",
        "Anacostia": "east of the Anacostia River, Frederick Douglass's Cedar Hill on the hilltop, Green Line at Anacostia, historic Black neighborhood, frame houses and modest rowhouses, sweeping skyline views from the bluffs, Anacostia Arts Center, working-class roots and disinvestment legacy, newer infill development pressure, civic and cultural reinvestment",
    },
    "philadelphia": {
        "Rittenhouse": "Rittenhouse Square park, Walnut Street luxury retail and boutiques, high-rise condo towers including The Laurel, brownstone side streets, Center City core, Market-Frankford and trolley access nearby, James Beard restaurant density, top price tier in the city, concierge and doorman buildings, finance and law professional register",
        "Fishtown": "Frankford Avenue restaurant and bar strip, La Colombe flagship roastery, former working-class fishing-village roots, Market-Frankford El at Girard, rowhouses with newer infill, music venues and craft breweries, creative-class and young-professional register, walkable to Northern Liberties, rapid-gentrification arc, tattoo-and-vinyl aesthetic",
        "Manayunk": "Main Street boutiques and bars, Schuylkill River and canal towpath, Schuylkill River Trail along the water, steep hill side streets, former mill-town industrial bones, regional rail at Manayunk station, twin and rowhouse stock, college-grad and young-professional register, bike-and-run culture, annual Manayunk Arts Festival",
        "Old City": "Independence Hall and Liberty Bell adjacent, cobblestone Elfreth's Alley, converted loft buildings in former warehouses, gallery First Fridays, Market-Frankford El at 2nd Street, federal and colonial-era brick, Delaware River waterfront, tourist-heavy daytime and bar-heavy weekends, design-and-architecture firm presence, historic district restrictions",
        "Northern Liberties": "Liberty Lands community park, 2nd Street corridor of bars and restaurants, Piazza redevelopment block, former industrial and brewing district, Market-Frankford El at Girard, rowhouses mixed with new mid-rise apartments, young-professional and creative register, walkable to Fishtown and Old City, annual 2nd Street Festival, infill construction throughout",
        "Society Hill": "federal and Georgian brick rowhouses, cobblestone and brick sidewalks, Head House Square, walkable to Independence National Historical Park, low-rise historic district with strict preservation rules, established old-Philadelphia register, leafy small parks and courtyards, PATCO and Market-Frankford access, professional and retiree mix, quiet and residential by Center City standards",
        "University City": "Penn and Drexel campuses, 30th Street Station regional rail hub, Market-Frankford El and trolley tunnels, Victorian twin houses west of campus, Clark Park and Baltimore Avenue corridor, academic and medical-center register, international graduate student presence, Cira Centre skyline, research-hospital workforce, mix of student rentals and family-owned blocks",
        "Queen Village": "Fabric Row along 4th Street, South Street commercial corridor, narrow trinity and rowhouses, Delaware River waterfront edge, historic Swedish-colony and Jewish-garment-district layers, walkable to Society Hill and Bella Vista, family and young-professional register, leafy residential blocks off the strip, no direct Metro but bus and walk to Market-Frankford",
    },
    "chicago": {
        "Gold Coast": "old-money pedigree, Astor Street brownstones, greystone mansions, Oak Street luxury retail, Rush Street steakhouses, Lake Shore Drive frontage, Oak Street Beach, Newberry Library, walkable, Red Line adjacent, pre-war high-rises beside Gilded Age townhomes",
        "Wicker Park": "indie music heritage, Six Corners at North-Milwaukee-Damen, vintage record shops, Milwaukee Avenue nightlife, Blue Line stop, graystone two-flats, converted lofts, hipster-to-yuppie creep, art-school crowd, cocktail bars and taquerias",
        "Lincoln Park": "DePaul University buzz, free Lincoln Park Zoo, lakefront trail, Victorian rowhouses, greystone single-families, Armitage boutiques, Brown and Red Line access, stroller-dense sidewalks, upper-middle professional, conservatory and ponds nearby",
        "Hyde Park": "University of Chicago gravity, Frank Lloyd Wright Robie House, Promontory Point lakefront, integrated middle-class enclave, Gothic limestone quads, brick six-flats, Metra Electric commute, bookstores and jazz clubs, intellectual register, south-side identity",
        "Pilsen": "Mexican-American heritage, 18th Street mural corridor, Pink Line station with Aztec station art, National Museum of Mexican Art, taquerias and panaderias, working-class roots, artist lofts in old warehouses, Czech-era brick two-flats, gentrification tension",
        "Logan Square": "boulevard medians and the Illinois Centennial Monument eagle column, Lula Cafe and farm-to-table dining, craft cocktail bars, Blue Line at the square, graystones and bungalows, young creative-class influx, Hispanic working-class undercurrent, walkable",
        "West Loop": "former meatpacking warehouses, Fulton Market food scene, Restaurant Row on Randolph Street, converted brick-and-timber lofts, Green Line and Pink Line access, tech and Google offices, Greektown edge, no-kids dense professional crowd, gallery walks",
        "Lakeview": "Wrigley Field and Cubs nights, Boystown LGBTQ corridor on Halsted, Southport Corridor boutiques, Red and Brown Line stops, three-flats and walk-ups, theater row, post-college and young-family mix, lakefront jogging path, sports-bar density",
    },
    "seattle": {
        "Capitol Hill": "LGBTQ stronghold, rainbow crosswalks at Pike and Pine, Cal Anderson Park, light rail station to SeaTac, indie music venues, brick apartment buildings and Craftsman walk-ups, queer nightlife, coffee-shop dense, young renters, dive bars beside cocktail rooms",
        "Ballard": "Scandinavian fishing-town roots, National Nordic Museum, Ballard Locks and the salmon ladder, Sunday farmers market on Ballard Avenue, craft brewery cluster, new mid-rise apartments over older bungalows, Burke-Gilman trail access, maritime working-class memory, family-friendly",
        "Fremont": "self-styled Center of the Universe, Fremont Troll under the Aurora Bridge, Lenin statue, Burke-Gilman Trail along the canal, Sunday flea market, Adobe and Google offices, quirky-creative register, bungalows and townhomes, walkable bridge to downtown",
        "University District": "University of Washington campus, The Ave commercial strip on University Way, Husky Stadium and the UW light rail station, Drumheller Fountain on Rainier Vista, student-rental brick walk-ups, cheap pho and bookstores, transient renter base, Craftsman houses on the side streets",
        "West Seattle": "Alki Beach skyline views, The Junction commercial core, King County Water Taxi to downtown, mid-century ramblers and view homes, ferry-and-bridge dependent, peninsula feel, family-oriented, Admiral District cafes, Lincoln Park trails",
        "South Lake Union": "Amazon headquarters and the Spheres, South Lake Union Streetcar, Lake Union Park and seaplane harbor, biotech and tech offices, glass-tower apartments, no historic stock, transient young-professional renters, Whole Foods and chain dining, post-2010 build-out",
        "Queen Anne": "Kerry Park Space Needle skyline view, ornate Queen Anne Victorians, Craftsman bungalows and Tudor revivals, steep hillside streets, Upper Queen Anne village shops, established professional and family register, walkable but car-reliant, quiet residential pockets above the city",
        "Wallingford": "Gas Works Park on Lake Union, Craftsman bungalows from the 1920s, 45th Street commercial strip, original Dick's Drive-In, Burke-Gilman Trail access, family streetcar-suburb stock, wide porches and mature trees, mellow middle-class register, walkable",
    },
    "denver": {
        "LoHi": "Lower Highland reinvention, Little Man Ice Cream giant milk-can, Linger rooftop in the old mortuary, pedestrian Highland Bridge to downtown, skyline views, modern townhomes beside Victorians, trendy restaurants and bars, young-professional density, post-industrial sheen",
        "RiNo": "River North Art District, dense mural walls along Larimer Street, craft brewery cluster, converted warehouse galleries, First Friday art walks, A Line commuter rail nearby, new live-work lofts, industrial brick stock, creative-class influx, rapid gentrification",
        "Wash Park": "165-acre Washington Park with two lakes, brick bungalows and Tudors, Old South Pearl Street shops and Sunday farmers market, Cherry Creek Trail access, jogger and cyclist culture, established upper-middle families, tree-lined streets, quiet residential register",
        "Cherry Creek": "Cherry Creek Shopping Center luxury mall, Cherry Creek North open-air boutiques, Cherry Creek Trail along the waterway, designer flagships, high-end condos and townhomes, mid-century ranches on side streets, fine-dining row, top Walk Scores, affluent and discreet",
        "Capitol Hill": "Cheesman Park green expanse, Molly Brown House Museum, Queen Anne and Richardsonian Romanesque mansions, Denver Square fourplexes, converted-mansion apartments, dense rental stock, young-renter and LGBTQ presence, walkable, eclectic socioeconomic mix",
        "Five Points": "historically Black neighborhood, Harlem of the West jazz legacy, Welton Street corridor, L Line light rail down the median, Victorian rowhouses and brick duplexes, juke-joint history, rapid recent gentrification, Five Points Jazz Festival, working-class memory beside new condos",
        "Highlands": "32nd Avenue Highland Square dining, Tennyson Street arts district, Victorian cottages and Denver Square brick houses, First Friday art walks, walkable village pockets, young-family register, Sloan's Lake access, mountain-view streets, older Italian and Hispanic roots",
        "Central Park": "former Stapleton airport site, master-planned post-2001 community, 80-acre Central Park, Shops at Northfield retail, front-porch new-traditional homes, wide tree-lined streets, young-family heavy, A Line commuter rail access, renamed in 2020, suburban density inside city limits",
    },
    "atlanta": {
        "Buckhead": "Lenox Square and Phipps Plaza luxury malls, designer flagships, estate-lot mansions on tree-lined streets, Ritz-Carlton and St. Regis residences, Chastain Park concerts, MARTA Red Line access, high-rise condo towers, old-money and new-money mix, the Beverly Hills of the East",
        "Midtown": "Piedmont Park 200-acre green space, High Museum of Art, Fox Theatre and Alliance Theatre, MARTA Red and Gold Line stations, Atlanta Botanical Garden, glass condo towers, Tech Square and Georgia Tech adjacency, dense walkable arts district, young-professional and LGBTQ presence",
        "Inman Park": "Atlanta's first planned suburb, ornate Victorian mansions, Krog Street Market in the old stove works, Eastside BeltLine Trail, annual Inman Park Festival and Tour of Homes, wraparound porches and turrets, MARTA Inman Park station, creative-class and design-firm crowd, walkable",
        "Virginia-Highland": "1920s streetcar-suburb stock, Craftsman bungalows and English Tudors, intersection of Virginia and North Highland Avenues, Atkins Park Restaurant since 1922, Summerfest, walkable shop-and-restaurant village, Piedmont Park edge, established middle-to-upper-middle families",
        "Decatur": "downtown Decatur Square and DeKalb Courthouse, MARTA Blue Line station under the plaza, Brick Store Pub and Kimball House dining, separate city in DeKalb County, bungalows and Craftsman houses, top-rated city schools, annual Decatur Book Festival heritage, walkable small-town feel, progressive register",
        "Old Fourth Ward": "Ponce City Market in the old Sears building, MLK birth home and Ebenezer Baptist Church, Eastside BeltLine Trail, Historic Fourth Ward Park, shotgun cottages beside new condos, post-BeltLine price surge, Black historical legacy, mixed-income tension, dense walkability",
        "Cabbagetown": "former Fulton Bag and Cotton Mill village, shotgun-cottage rows, narrow mill-town streets, Krog Street Tunnel street art, Oakland Cemetery edge, Appalachian working-class roots, artist-musician revival from the 1980s, eclectic and creative register, tight-knit and walkable",
        "Castleberry Hill": "converted 19th-century brick warehouses, residential lofts, art-gallery cluster on Walker and Peters, second-Friday Art Stroll, Mercedes-Benz Stadium adjacency, MARTA Garnett station, small 40-acre historic arts district, Black artist community, industrial-creative aesthetic",
    },
    "portland": {
        "Pearl District": "former warehouse district, converted brick lofts, glass high-rises, Powell's Books-adjacent, Portland Streetcar NS Line, Jamison Square, Tanner Springs Park, First Thursday gallery walk, design showrooms, upscale grocers, walkable, professional and downsizer mix",
        "NW Nob Hill": "tree-lined NW 23rd Avenue, victorian and edwardian houses converted to boutiques, Alphabet District streets, sidewalk cafes, Forest Park trailheads nearby, streetcar-served, established old-money quiet, upper-middle-class professionals",
        "Alberta Arts District": "NE Alberta Street corridor, Last Thursday art walk, dense mural concentration, craftsman bungalows and old portland foursquares, Black cultural roots, indie galleries, food carts, taprooms, bikeable, gentrifying working-class blocks",
        "Hawthorne": "SE Hawthorne Boulevard, bohemian retail strip, vintage shops, Bagdad Theater, Powell's Books on Hawthorne, craftsman bungalows and foursquares, ladd's addition nearby, walker's paradise, bike lanes, eclectic mid-tier",
        "Mississippi": "N Mississippi Avenue, historic Black Boise-Eliot roots, Mississippi Studios live music, Prost beer hall food carts, independent shops, Por Que No taqueria, craftsman bungalows, walkable corridor, rapidly gentrified working-class",
        "Sellwood": "Antique Row on SE 13th, riverfront on the Willamette, Oaks Amusement Park, craftsman bungalows and early-century cottages, walkable village feel, small-town quiet, family-oriented, established middle-class",
        "Goose Hollow": "Providence Park stadium-adjacent, MAX Red and Blue Line stops, mid-century co-ops and contemporary towers, modest bungalows beside victorian mansions, Washington Park nearby, Timbers fans, downtown-edge convenience, mixed-income",
        "St Johns": "St Johns Bridge suspension cables, Cathedral Park beneath the arches, small-town main street, locally-owned pubs and cafes, modest bungalows and cottages, industrial Willamette riverfront, blue-collar roots, last-affordable North Portland",
    },
    "phoenix": {
        "Arcadia": "citrus-grove origins, flood-irrigated quarter-acre lots, mature tree canopy, Camelback Mountain views, Echo Canyon trailhead, ranch homes and contemporary rebuilds, LGO and Postino on 40th Street, Scottsdale schools, luxury family",
        "Biltmore": "Arizona Biltmore resort grounds, two championship golf courses, Wrigley Mansion on the hill, Biltmore Fashion Park retail, mid-rise luxury condos, Camelback Corridor offices, manicured landscaping, high-end professional and retiree",
        "Roosevelt Row": "downtown Phoenix arts corridor, dense mural walls, First Friday Art Walk, rehabbed early-century bungalows beside new mid-rises, light rail-adjacent, indie galleries, craft taprooms, walkable, creative-class and renter-heavy",
        "Camelback East": "24th-to-44th Street stretch, flood-irrigated lawns, mid-century ranch homes on deep lots, Camelback Corridor offices, Phoenix Mountain Preserve access, established green pockets, mix of single-family and multifamily, upper-middle-class",
        "Encanto": "Encanto-Palmcroft historic district, 222-acre Encanto Park, winding garden-suburb streets, spanish colonial and tudor revival homes, 1920s bungalows, central Phoenix quiet, professional preservationist, downtown-adjacent",
        "Coronado": "Phoenix's first planned subdivision, craftsman bungalows and tudor revival cottages, front-porch culture, midtown location, light rail-accessible, artsy young homeowners, working-class roots, gentrifying historic",
        "North Phoenix": "Desert Ridge master-planned hub, Desert Ridge Marketplace, High Street shops and restaurants, Reach 11 trails, single-family subdivisions and townhomes, Wildfire Golf Club, top-rated schools, freeway-connected suburban family",
        "Ahwatukee": "South Mountain foothills, sprawling municipal park trails, master-planned subdivisions, golf courses, scenic desert views, stucco family homes, strong schools, quieter southwest valley, established suburban family",
    },
    "dallas": {
        "Highland Park": "incorporated town within Dallas, Highland Park Village luxury retail, tudor and mediterranean estates, georgian mansions, Lakeside Park azaleas, top-ranked HPISD schools, mature canopy and creeks, multimillion-dollar old-money",
        "Uptown": "McKinney Avenue Trolley free streetcar, Katy Trail rail-to-trail, Klyde Warren Park, luxury mid-rise and high-rise apartments, walkable bar and restaurant rows, Arts District-adjacent, young professional renter",
        "Bishop Arts": "North Oak Cliff core, no-chain merchant association, independent boutiques and bookstores, Spinster Records, Texas Theater nearby, 1920s storefronts on Bishop and 7th, walkable, creative-class, gentrified Latino corridor",
        "M-Streets": "Greenland Hills conservation district, 1920s tudor revival cottages, brick facades, century-old oak and pecan canopy, McCommas and Mercedes named streets, Lower Greenville bar and restaurant rows, young-family upper-middle-class",
        "Lakewood": "White Rock Lake trails, tudor and spanish eclectic homes, mid-century ranches, mature tree canopy, Lakewood Country Club, Arboretum-adjacent, walkable cottage blocks, established family-oriented East Dallas",
        "Deep Ellum": "historic Black music corridor, Blind Lemon Jefferson legacy, warehouse-loft conversions, over a hundred murals, Bomb Factory and Trees venues, Elm Street spine, tattoo parlors and taprooms, creative renter and nightlife",
        "Knox/Henderson": "Knox Street luxury retail, RH Dallas Gallery, Henderson Avenue eclectic boutiques, Katy Trail access, mix of townhomes and low-rise apartments, walkable, between Highland Park and Uptown, upscale young professional",
        "Oak Cliff": "Jefferson Boulevard Latino corridor, taquerias and quinceanera shops, Kessler Park rolling hills, Texas Theater landmark, Dilbeck and Williams-designed cottages, Bishop Arts-adjacent, diverse working-class blending with newer creative-class",
    },
}


CITY_DISPLAY_NAMES = {
    "sf": "San Francisco", "boston": "Boston", "nyc": "New York",
    "dc": "Washington DC", "philadelphia": "Philadelphia",
    "chicago": "Chicago", "seattle": "Seattle", "denver": "Denver",
    "atlanta": "Atlanta", "portland": "Portland", "phoenix": "Phoenix",
    "dallas": "Dallas",
}


def get_submarket_hint(city: str, submarket: str) -> str:
    """City-aware submarket hint lookup. Falls back to a generic descriptor
    when (city, submarket) is not pre-curated, so the pipeline degrades
    gracefully on novel submarket strings.
    """
    city_map = SUBMARKET_HINTS.get(city, {})
    if submarket in city_map:
        return city_map[submarket]
    display = CITY_DISPLAY_NAMES.get(city, city)
    return f"the {submarket} neighborhood of {display}"

_OUTPUT_INSTR = (
    "Return ONLY a single JSON object with this exact schema and no other "
    "prose, code fences, or commentary:\n"
    "{\n"
    '  "rewritten_text": "<full rewritten listing description, plain text>",\n'
    '  "preserved_slots": {<copy of the slot-fact JSON you were given verbatim>}\n'
    "}\n"
    "If you cannot preserve a fact verbatim, set rewritten_text to the empty "
    "string and explain nothing — leave preserved_slots empty too."
)

_VERIFY_STEP = (
    "Before writing the final JSON, internally walk through every numeric "
    "fact in the slot-fact JSON and confirm it appears with the SAME value in "
    "your rewrite. Do not include this reasoning in your output — only the "
    "final JSON object."
)

_QUALITY_CRITERIA = (
    "QUALITY CRITERIA — a successful counterfactual rewrite:\n"
    "  - Numeric facts: every value in the slot-fact JSON appears unchanged.\n"
    "  - Style shift: a knowledgeable local could identify the target submarket\n"
    "    from the rewrite alone, without seeing the slot-fact JSON.\n"
    "  - Length: within ±20% of the original word count.\n"
    "  - Register: matches the polish of the original (MLS-formal vs casual).\n"
    "  - No double-claiming: do not assert prestige cues that were not in the\n"
    "    original. The rewrite should plausibly substitute for the original,\n"
    "    not upsell it."
)

_PITFALLS = (
    "COMMON FAILURE MODES TO AVOID:\n"
    "  - Generic openers (\"In the heart of [neighborhood]...\") — these are\n"
    "    tells of LLM-generated copy and dilute the submarket signal.\n"
    "  - Inventing transit lines, parks, schools, or businesses that do not\n"
    "    exist near the target submarket.\n"
    "  - Substituting a numeric fact (\"3 bedrooms\" → \"spacious bedrooms\")\n"
    "    to avoid repeating the slot — preserve the number explicitly.\n"
    "  - Adding new amenities (gym, pool, doorman) not in the original.\n"
    "  - Reusing landmark names from the original (Dolores Park, Crissy\n"
    "    Field, etc.) when the rewrite targets a different submarket."
)

_VERIFY_CHECKLIST = (
    "VERIFICATION CHECKLIST (run mentally before output):\n"
    "  1. Read each entry in the slot-fact JSON. For each entry, find the\n"
    "     identical value in your rewrite. If any is missing, REWRITE before\n"
    "     emitting.\n"
    "  2. Scan for any neighborhood name or street from the original. If\n"
    "     found, replace with a target-submarket equivalent (or remove for\n"
    "     style_stripped).\n"
    "  3. Confirm the rewrite would be plausible from a real-estate agent\n"
    "     working that neighborhood (or a flat MLS feed for style_stripped).\n"
    "     If it reads as foreign, revise the vocabulary."
)

_SWAP_EXAMPLE = (
    "WORKED EXAMPLE (Mission District → Pacific Heights swap):\n"
    "  Slot-fact JSON: {\"bedrooms\": 2, \"bathrooms\": 1, \"sqft\": 1100, "
    "\"year_built\": 1908, \"parking\": null}\n"
    "  Original: \"Charming 2-bedroom Victorian flat steps from Dolores Park "
    "and 24th Street taquerias. 1100 sqft, 1 bath, original 1908 millwork. "
    "Walk to BART. Mission's vibrant cafe scene at your door.\"\n"
    "  Good rewrite: \"Stately 2-bedroom Edwardian flat just off Lyon Street, "
    "walking distance to Fillmore Street boutiques. 1100 sqft, 1 bath, "
    "original 1908 millwork. Sweeping Bay views from the parlor floor. "
    "Refined Pacific Heights address with mature street trees.\"\n"
    "  Why it works: every slot-fact value preserved verbatim (2, 1, 1100, "
    "1908); Mission landmarks (Dolores Park, 24th Street, BART) replaced\n"
    "  with Pacific Heights equivalents (Lyon Street, Fillmore Street, Bay\n"
    "  views); register elevated to match the destination submarket\n"
    "  without inventing facts that were not in the original.\n"
    "  Bad rewrite (REJECT): \"In the heart of prestigious Pacific Heights, "
    "this stunning 2-bedroom Victorian masterpiece offers Dolores Park "
    "views, 1100+ sqft, with luxury parking included.\" Why it fails: "
    "generic opener, retains \"Dolores Park\" from origin submarket, "
    "embellishes square footage with \"+\", invents parking that the "
    "slot-fact JSON marks as null, and asserts unsupported prestige."
)

_STRIPPED_EXAMPLE = (
    "WORKED EXAMPLE (style_stripped):\n"
    "  Slot-fact JSON: {\"bedrooms\": 2, \"bathrooms\": 1, \"sqft\": 1100, "
    "\"year_built\": 1908, \"parking\": null}\n"
    "  Original: \"Charming 2-bedroom Victorian flat steps from Dolores Park "
    "and 24th Street taquerias. 1100 sqft, 1 bath, original 1908 millwork. "
    "Walk to BART. Mission's vibrant cafe scene at your door.\"\n"
    "  Good rewrite: \"2-bedroom flat. 1100 sqft. 1 bath. Built 1908. "
    "Original millwork. No parking. Single-level layout above ground floor.\"\n"
    "  Why it works: every slot-fact value preserved verbatim (2, 1, 1100, "
    "1908); all neighborhood references (Dolores Park, 24th Street, BART,\n"
    "  Mission), all aspirational lexicon (\"charming\", \"vibrant\"), and\n"
    "  all geographic cues removed; tone reduced to MLS-style enumeration\n"
    "  without inventing new facts.\n"
    "  Bad rewrite (REJECT): \"Beautifully maintained 2-bedroom residence "
    "in a desirable urban setting. 1100+ sqft of charming living space. "
    "1 bath. Built in 1908. Excellent transit access. Move-in ready.\" "
    "Why it fails: retains aspirational lexicon (\"beautifully\", "
    "\"charming\", \"desirable\"), implies geography (\"urban\", "
    "\"transit access\"), embellishes square footage with \"+\", and "
    "asserts prestige cues (\"move-in ready\") not in the original."
)


def _format_slots(slot_dict: dict[str, Optional[float]]) -> str:
    """JSON-serialize slots with None preserved as null."""
    return json.dumps({k: v for k, v in slot_dict.items()}, indent=2)



def style_swap_system() -> str:
    """Constant instruction template for style_swap. Cacheable.

    Does NOT include the target submarket name, hint, slot dict, or original
    text — those vary per call and live in the user message.
    """
    return (
        "You are a real-estate copywriter producing a counterfactual listing "
        "rewrite for an academic causal-inference experiment.\n\n"
        "GENERAL TASK: Rewrite a property listing as if the property were "
        "located in a different submarket of San Francisco. Preserve every "
        "numeric property fact verbatim; replace every submarket-evocative "
        "cue with an equivalent for the target submarket.\n\n"
        "PRESERVE EXACTLY (must appear with identical numeric values in the rewrite):\n"
        "  - bedroom count\n"
        "  - bathroom count\n"
        "  - square footage\n"
        "  - year built\n"
        "  - lot size (if present)\n"
        "  - parking / garage capacity (if present)\n"
        "  - number of stories / levels (if present)\n\n"
        "CHANGE (this is the counterfactual treatment):\n"
        "  - all neighborhood names, landmarks, and street references\n"
        "  - all submarket-evocative lexicon (vibe words, cultural cues, view claims)\n"
        "  - implied price tier and prestige cues\n"
        "  - the target submarket should be unambiguous to a local reader\n\n"
        "DO NOT:\n"
        "  - invent new numeric facts\n"
        "  - change any fact in the slot-fact JSON the user provides\n"
        "  - retain any landmark or street name from the original\n\n"
        f"{_QUALITY_CRITERIA}\n\n"
        f"{_PITFALLS}\n\n"
        f"{_SWAP_EXAMPLE}\n\n"
        f"{_VERIFY_CHECKLIST}\n\n"
        f"{_VERIFY_STEP}\n\n"
        f"{_OUTPUT_INSTR}"
    )


def style_swap_user(
    target_submarket: str,
    original_text: str,
    slot_dict: dict[str, Optional[float]],
    city: str = "sf",
) -> str:
    """Variable per-call content for style_swap. NOT cacheable."""
    hint = get_submarket_hint(city, target_submarket)
    city_display = CITY_DISPLAY_NAMES.get(city, city)
    return (
        f"TARGET SUBMARKET: {target_submarket}, {city_display}.\n\n"
        f"TARGET SUBMARKET STYLE NOTES (use this lexicon and vibe):\n  {hint}\n\n"
        f"SLOT-FACT JSON (must be preserved verbatim):\n{_format_slots(slot_dict)}\n\n"
        f"ORIGINAL LISTING:\n\"\"\"\n{original_text}\n\"\"\""
    )


def style_swap_blocks(
    target_submarket: str,
    original_text: str,
    slot_dict: dict[str, Optional[float]],
    city: str = "sf",
) -> dict[str, str]:
    return {
        "system": style_swap_system(),
        "user": style_swap_user(target_submarket, original_text, slot_dict, city=city),
    }


def style_stripped_system() -> str:
    """Constant instruction template for style_stripped. Cacheable."""
    return (
        "You are a real-estate copywriter producing a NEUTRALIZED counterfactual "
        "listing rewrite for an academic causal-inference experiment.\n\n"
        "GENERAL TASK: Rewrite a property listing in a flat, neutral, fact-"
        "forward tone that strips ALL neighborhood-evocative language while "
        "preserving every numeric property fact.\n\n"
        "PRESERVE EXACTLY (must appear with identical numeric values in the rewrite):\n"
        "  - bedroom count, bathroom count, square footage\n"
        "  - year built, lot size, parking capacity, story count (if present)\n"
        "  - structural facts (room layout, construction features)\n\n"
        "STRIP / REMOVE (this is the counterfactual treatment):\n"
        "  - all neighborhood names and submarket references\n"
        "  - all landmark, street, park, and transit-line names\n"
        "  - all aspirational lifestyle lexicon (\"vibrant\", \"prestigious\", "
        "\"sun-drenched\", \"coveted\", etc.)\n"
        "  - all view claims that imply geography (\"ocean views\", \"Bay views\")\n"
        "  - all cultural / demographic cues\n"
        "  - all price-tier signals (\"luxury\", \"entry-level\", "
        "\"investment opportunity\")\n\n"
        "TONE: dry MLS-style enumeration of physical specifications. Short "
        "sentences. No adjectives that signal neighborhood prestige.\n\n"
        "DO NOT:\n"
        "  - mention San Francisco or any neighborhood\n"
        "  - invent new numeric facts\n"
        "  - change any fact in the slot-fact JSON the user provides\n\n"
        f"{_QUALITY_CRITERIA}\n\n"
        f"{_PITFALLS}\n\n"
        f"{_STRIPPED_EXAMPLE}\n\n"
        f"{_VERIFY_CHECKLIST}\n\n"
        f"{_VERIFY_STEP}\n\n"
        f"{_OUTPUT_INSTR}"
    )


def style_stripped_user(
    original_text: str,
    slot_dict: dict[str, Optional[float]],
) -> str:
    """Variable per-call content for style_stripped. NOT cacheable."""
    return (
        f"SLOT-FACT JSON (must be preserved verbatim):\n{_format_slots(slot_dict)}\n\n"
        f"ORIGINAL LISTING:\n\"\"\"\n{original_text}\n\"\"\""
    )


def style_stripped_blocks(
    original_text: str,
    slot_dict: dict[str, Optional[float]],
) -> dict[str, str]:
    return {
        "system": style_stripped_system(),
        "user": style_stripped_user(original_text, slot_dict),
    }



def style_swap_prompt(
    target_submarket: str,
    original_text: str,
    slot_dict: dict[str, Optional[float]],
    city: str = "sf",
) -> str:
    """Single-string form, retained for backward compat. Prefer style_swap_blocks."""
    blocks = style_swap_blocks(target_submarket, original_text, slot_dict, city=city)
    return f"{blocks['system']}\n\n{blocks['user']}"


def style_stripped_prompt(
    original_text: str,
    slot_dict: dict[str, Optional[float]],
) -> str:
    """Single-string form, retained for backward compat. Prefer style_stripped_blocks."""
    blocks = style_stripped_blocks(original_text, slot_dict)
    return f"{blocks['system']}\n\n{blocks['user']}"
