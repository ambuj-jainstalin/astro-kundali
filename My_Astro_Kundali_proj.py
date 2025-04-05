from datetime import datetime,timedelta
import os
import time
import streamlit as st
import dotenv
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
import swisseph as swe
import json
import requests
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import plotly.graph_objects as go
import plotly
from matplotlib.patches import Polygon
import svgwrite
from svgwrite import cm, mm
import math
import numpy as np
import matplotlib.pyplot as plt
from click import style
from matplotlib.patches import Polygon
from pytz import timezone


load_dotenv()


# ✅ Set Ephemeris Path
swe.set_ephe_path("/path/to/ephemeris/")
# swe_set_jpl_file("");

# ✅ Initialize Session State
if "user_details" not in st.session_state:
    st.session_state.user_details = None
if "astro_data" not in st.session_state:
    st.session_state.astro_data = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()
if "first_message_sent" not in st.session_state:
    st.session_state.first_message_sent = False

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GOOGLE_API_KEY"),temperature=0.7,top_p=0.85)
if "conversation" not in st.session_state:
    system_prompt="""You are an AI Astrologer. Follow these guidelines while responding:
    1. Focus Only on Astrology: Do not discuss topics unrelated to astrology. If a user asks, politely steer the conversation back.
    2. No Unnecessary Preface: Directly provide insights without generic phrases like "As an AI, I can help you with..." or "based on information provided"
    3. Use Astrological Calculations: Ensure accurate calculations for Kundali, planetary positions, houses, and transits.
    4. Provide Specific Predictions: Use Vedic astrology principles to give personalized readings.
    5. No Repetitive or Vague Responses: Avoid over-explaining concepts the user already knows.
    6. Follow Traditional and Modern Astrology: Use a blend of classical Vedic astrology and computational analysis where required.
    7. Handle Birth Details Carefully: Always refer to the user’s DOB, time, and place before making predictions.
    8. Be Concise Yet Informative: Provide detailed yet digestible responses, avoiding unnecessary fluff.
    9. Use Simple Language: Make astrology accessible to all users.
    10. Do Not Generate Random Predictions: Base insights strictly on astrological data.
    11. First Provide Snapsot like , Birth Details, Sun Sign, Moon Sign, Nakshatra, Day of Birth, Yoni etcs"""
    st.session_state.memory.chat_memory.add_user_message(system_prompt)
    st.session_state.conversation = ConversationChain(llm=llm, memory=st.session_state.memory)


# ✅ Function to Calculate Astrology Chart

# First install: pip install pykundli




# Kundali data
# Create Kundali Chart

def display_kundli_chart(astro_data):
    fig = create_chart(astro_data)  # Modified create_chart to return figure
    st.pyplot(fig)

def create_chart(astro_data):
    # kundali_data = {
    #     'Sun': {'Degree': 153.2581, 'Sign': 'Virgo', 'Retrograde': False, 'House': 8},
    #     'Moon': {'Degree': 292.9164, 'Sign': 'Capricorn', 'Retrograde': False, 'House': 12},
    #     'Mercury': {'Degree': 142.1335, 'Sign': 'Leo', 'Retrograde': False, 'House': 7},
    #     'Venus': {'Degree': 118.2022, 'Sign': 'Cancer', 'Retrograde': False, 'House': 6},
    #     'Mars': {'Degree': 168.7229, 'Sign': 'Virgo', 'Retrograde': False, 'House': 8},
    #     'Jupiter': {'Degree': 127.9802, 'Sign': 'Leo', 'Retrograde': False, 'House': 7},
    #     'Saturn': {'Degree': 276.6245, 'Sign': 'Capricorn', 'Retrograde': False, 'House': 12},
    #     'Rahu': {'Degree': 261.4899, 'Sign': 'Sagittarius', 'Retrograde': False, 'House': 11},
    #     'Uranus': {'Degree': 256.0901, 'Sign': 'Sagittarius', 'Retrograde': False, 'House': 11},
    #     'Neptune': {'Degree': 260.2516, 'Sign': 'Sagittarius', 'Retrograde': False, 'House': 11},
    #     'Pluto': {'Degree': 204.605, 'Sign': 'Libra', 'Retrograde': False, 'House': 9},
    #     'Ascendant': {'Degree': 326.81, 'Sign': 'Aquarius'},
    #     'Moon Nakshatra': 'Shravana'
    # }

    # Planetary abbreviations and colors
    planet_abbr = {
        'Sun': 'Su',
        'Moon': 'Mo',
        'Mercury': 'Me',
        'Venus': 'Ve',
        'Mars': 'Ma',
        'Jupiter': 'Ju',
        'Saturn': 'Sa',
        'Rahu': 'Ra',
        'Ketu': 'Ke',
        'Uranus': 'Ur',
        'Neptune': 'Ne',
        'Pluto': 'Pl',
        'Ascendant': 'La'
    }

    planet_colors = {
        'Sun': 'red',
        'Moon': 'black',
        'Mercury': 'blue',
        'Venus': 'green',
        'Mars': 'darkgreen',
        'Jupiter': 'purple',
        'Saturn': 'black',
        'Rahu': 'red',
        'Ketu': 'brown',
        'Uranus': 'red',
        'Neptune': 'navy',
        'Pluto': 'black',
        'Ascendant': 'black'
    }

    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
    ax.set_aspect('equal')
    ax.axis('off')
    ref_size = min(fig.get_size_inches()) * fig.dpi

    # Original rectangle and lines
    # rectangle = np.array([[0, 0], [12, 0], [12, 8], [0, 8], [0, 0]])
    # ax.plot(rectangle[:, 0], rectangle[:, 1], color='red', linewidth=2)
    # Calculate scaling factor based on figure width
    original_width = 12.0  # Original reference width in inches
    current_width = fig.get_size_inches()[0]
    scaling_factor = current_width / original_width
    # Key intersection points
    A, B, C, D = (3, 2), (9, 2), (3, 6), (9, 6)
    center = (6, 4)

    # Define all shapes with HOUSE NUMBERS
    house_assignments = [
        # Shape coordinates (normalized)       Label position (normalized)
        (np.array([[0.25, 0.75], [0.5, 1.0], [0.75, 0.75], [0.5, 0.5]]), (0.5, 0.83)),  # House 1
        (np.array([[0.0, 1.0], [0.25, 0.75], [0.5, 1.0]]), (0.25, 0.92)),  # House 2
        (np.array([[0.0, 1.0], [0.0, 0.5], [0.25, 0.75]]), (0.10, 0.80)),  # House 3
        (np.array([[0.0, 0.5], [0.25, 0.25], [0.5, 0.5], [0.25, 0.75]]), (0.25, 0.65)),  # House 4
        (np.array([[0.0, 0.0], [0.0, 0.5], [0.25, 0.25]]), (0.10, 0.35)),  # House 5
        (np.array([[0.0, 0.0], [0.25, 0.25], [0.5, 0.0]]), (0.28, 0.20)),  # House 6
        (np.array([[0.25, 0.25], [0.5, 0.5], [0.75, 0.25], [0.5, 0.0]]), (0.5, 0.35)),  # House 7
        (np.array([[0.5, 0.0], [0.75, 0.25], [1.0, 0.0]]), (0.72, 0.20)),  # House 8
        (np.array([[1.0, 0.0], [1.0, 0.5], [0.75, 0.25]]), (0.89, 0.33)),  # House 9
        (np.array([[1.0, 0.5], [0.75, 0.75], [0.5, 0.5], [0.75, 0.25]]), (0.70, 0.65)),  # House 10
        (np.array([[1.0, 1.0], [1.0, 0.5], [0.75, 0.75]]), (0.89, 0.78)),  # House 11
        (np.array([[0.5, 1.0], [0.75, 0.75], [1.0, 1.0]]), (0.75, 0.92))  # House 12
    ]
    for shape, label_pos in house_assignments:
        shape[:, 0] /= 12  # Normalize X coordinates
        shape[:, 1] /= 8  # Normalize Y coordinates
        label_pos = (label_pos[0] / 12, label_pos[1] / 8)
    # Create house data mapping
    house_contents = {i + 1: [] for i in range(12)}
    for planet, data in astro_data.items():
        if planet == 'Ascendant':
            house_contents[1].append(('Ascendant', data['Sign']))
        elif planet != 'Moon Nakshatra':
            house_contents[data.get('House', 0)].append((planet, data['Sign']))
    # print(house_contents)
    house_patches = []
    text_objects = []
    # Draw and label all houses with dynamic sizing
    for idx, (shape_data, label_pos) in enumerate(house_assignments):
        house_num = idx + 1
        patch = Polygon(shape_data, closed=True, edgecolor='red', fill=False, lw=1)
        ax.add_patch(patch)
        house_patches.append(patch)

        # Add house number
        t = ax.text(label_pos[0], label_pos[1], str(house_num),
                    transform=ax.transAxes,
                    ha='center', va='center',
                    fontsize=12, fontweight='bold', color='darkblue')
        text_objects.append(t)

        # Add planet positions
        if house_num in house_contents:
            y_offset = label_pos[1] - 0.03
            for i, (planet, sign) in enumerate(house_contents[house_num]):
                t = ax.text(label_pos[0], y_offset - i * 0.03,
                            f"{planet_abbr.get(planet, planet[:2])}({sign[:3]})",
                            transform=ax.transAxes,
                            ha='center', va='top',
                            fontsize=8, color=planet_colors.get(planet, 'black'),
                            fontweight='bold')
                text_objects.append(t)

    # Function to adjust font sizes on resize
    def on_resize(event):
        fig = event.canvas.figure
        ax = fig.axes[0]
        new_size = min(fig.get_size_inches()) * fig.dpi

        # Update font sizes based on new size
        base_font = new_size / 80
        for text in text_objects:
            if text.get_text().isdigit():  # House numbers
                text.set_fontsize(base_font * 1.2)
            else:  # Planet labels
                text.set_fontsize(base_font * 0.8)

    # Redraw original lines on top
    ax.plot([0, 12], [0, 8], color='red')  # Main diagonals
    ax.plot([0, 12], [8, 0], color='red')
    ax.plot([0, 6], [4, 0], color='red')  # Middle connections
    ax.plot([0, 6], [4, 8], color='red')
    ax.plot([12, 6], [4, 0], color='red')
    ax.plot([12, 6], [4, 8], color='red')
    ax.plot([0, 0], [0, 8], color='red')
    ax.plot([0, 0], [0, 8], color='red')  # Left side
    ax.plot([0, 12], [8, 8], color='red')  # Top side
    ax.plot([12, 12], [8, 0], color='red')  # Right side
    ax.plot([12, 0], [0, 0], color='red')

    # Add legend for planetary symbols
    # legend_elements = []
    # for planet, abbr in planet_abbr.items():
    #     if planet in planet_colors and planet != 'Moon Nakshatra':
    #         legend_elements.append(plt.Line2D([0], [0],
    #                                           marker='o',
    #                                           color='w',
    #                                           label=f"{abbr}: {planet}",
    #                                           markerfacecolor=planet_colors[planet],
    #                                           markersize=10))

    # ax.legend(handles=legend_elements,
    #           title='Planetary Symbols',
    #           bbox_to_anchor=(1.25, 1),
    #           loc='upper left')

    plt.title('Lagna Chart', fontsize=16, fontweight='bold')
    fig.canvas.mpl_connect('resize_event', on_resize)
    return fig
# def create_vedic_kundli(astro_data):


def calculate_chart(dob, time_of_birth, place_of_birth, ayanamsa: str = 'ay_lahiri'):
    # Convert to Julian Date

    SWE_AYANAMSA = {
        "ay_fagan_bradley": 0,
        "ay_lahiri": 1,
        "ay_deluce": 2,
        "ay_raman": 3,
        "ay_krishnamurti": 5,
        "ay_sassanian": 16,
        "ay_aldebaran_15tau": 14,
        "ay_galcenter_5sag": 17
    }

    ist_datetime = datetime.combine(dob, time_of_birth)
    utc_datetime = ist_datetime - timedelta(hours=5, minutes=30)

    # Convert to Julian Day (UTC)
    juld = swe.utc_to_jd(
        utc_datetime.year, utc_datetime.month, utc_datetime.day,
        utc_datetime.hour, utc_datetime.minute, utc_datetime.second,
        swe.GREG_CAL)[1]
    swe.set_sid_mode(SWE_AYANAMSA[ayanamsa.lower()], 0, 0)  # Set the Ayanamsa # Set the Ayanamsa
    flags = swe.FLG_SWIEPH + swe.FLG_SPEED + swe.FLG_SIDEREAL


    # Geocode birthplace using OpenCage
    oc_api_key = os.getenv("OPENCAGE_API_KEY")
    url = f"https://api.opencagedata.com/geocode/v1/json?q={place_of_birth}&key={oc_api_key}"

    try:
        response = requests.get(url)
        data = response.json()

        if data['results'] and len(data['results']) > 0:
            lat = data['results'][0]['geometry']['lat']
            lon = data['results'][0]['geometry']['lng']
            # lat=27.1767
            # lon=78.0081
            st.session_state.location_data = {"lat": lat, "lon": lon, "place": place_of_birth}
            # print(lat, lon)
        else:
            st.error(f"Could not find coordinates for {place_of_birth}")
            return None
    except Exception as e:
        st.error(f"Error getting location data: {e}")
        return None

    # Planets & Zodiac Signs
    SWE_AYANAMSA = {
        "ay_fagan_bradley": 0,
        "ay_lahiri": 1,
        "ay_deluce": 2,
        "ay_raman": 3,
        "ay_krishnamurti": 5,
        "ay_sassanian": 16,
        "ay_aldebaran_15tau": 14,
        "ay_galcenter_5sag": 17
    }

    PLANETS = {
        "Sun": swe.SUN, "Moon": swe.MOON, "Mercury": swe.MERCURY, "Venus": swe.VENUS, "Mars": swe.MARS,
        "Jupiter": swe.JUPITER, "Saturn": swe.SATURN, "Rahu": swe.MEAN_NODE,
        "Uranus": swe.URANUS, "Neptune": swe.NEPTUNE, "Pluto": swe.PLUTO
    }
    ZODIAC_SIGNS= ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
                    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
    nakshatras = ["Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra", "Punarvasu",
                  "Pushya", "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni", "Hasta", "Chitra",
                  "Swati", "Vishakha", "Anuradha", "Jyeshtha", "Mula", "Purva Ashadha", "Uttara Ashadha",
                  "Shravana", "Dhanishta", "Shatabhisha", "Purva Bhadrapada", "Uttara Bhadrapada", "Revati"]

    # Set sidereal mode for Vedic calculations
    output = {}
    try:
        swe.set_ephe_path('/path/to/ephemeris/')

        # juld = swe.utc_to_jd(year_utc, month_utc, day_utc, hour_utc, minute_utc, second_utc, swe.GREG_CAL)[1]
        swe.set_sid_mode(SWE_AYANAMSA[ayanamsa.lower()], 0, 0)  # Set the Ayanamsa
        flags = swe.FLG_SWIEPH + swe.FLG_SPEED + swe.FLG_SIDEREAL

        # Calculate houses and ascendant
        cusps, ascmc = swe.houses_ex(juld, lat, lon, b'B', flags)
        ascendant_lon = ascmc[0]
        output['Ascendant'] = {
            'Degree': ascendant_lon % 30,
            'Sign': ZODIAC_SIGNS[int(ascendant_lon / 30)],
        }
        print(ascendant_lon)

        # Calculate planet positions
        for planet_name, planet_code in PLANETS.items():
            xx, ret = swe.calc_ut(juld, planet_code, flags)
            longitude = xx[0]
            sign_index = int(longitude / 30) % 12
            output[planet_name.capitalize()] = {
                'Degree': longitude % 30,
                'Sign': ZODIAC_SIGNS[sign_index],
                'Retrograde': xx[3] < 0,
                'raw_longitude': longitude
            }

        # Calculate Rahu and Ketu
        rahu_lon = output['Rahu']['raw_longitude']
        ketu_lon = swe.degnorm(rahu_lon + 180)
        ketu_sign_index = int(ketu_lon / 30) % 12
        output['Ketu'] = {
            'Degree': ketu_lon % 30,
            'Sign': ZODIAC_SIGNS[ketu_sign_index],
            'Retrograde': output['Rahu']['Retrograde'],
            'raw_longitude': ketu_lon
        }
        del output['Rahu']['raw_longitude']
        del output['Ketu']['raw_longitude']

        # Calculate Moon Nakshatra (crude approximation)
        if 'Moon' in output:
            moon_lon = output['Moon']['raw_longitude']
            nakshatra_number = int((moon_lon * 3) / 40) % 27 + 1
            nakshatras = [
                "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashirsha", "Ardra",
                "Punarvasu", "Pushya", "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni",
                "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha",
                "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana", "Dhanishta", "Shatabhisha",
                "Purva Bhadrapada", "Uttara Bhadrapada", "Revati"
            ]
            output['Moon Nakshatra'] = nakshatras[nakshatra_number - 1]
            del output['Moon']['raw_longitude']

        # Assign houses (Lagna Chart)
        houses = [None] * 12
        asc_sign_num = int(ascendant_lon / 30)
        for i in range(12):
            sign_index = (asc_sign_num + i) % 12
            houses[i] = {'sign_index': sign_index, 'planets': []}

        for planet_name, data in output.items():
            if planet_name not in ['Ascendant', 'Moon Nakshatra']:
                sign_index = ZODIAC_SIGNS.index(data['Sign'])
                house_index = (sign_index - asc_sign_num) % 12
                output[planet_name]['House'] = house_index + 1



    except Exception as e:
        print(f"An error occurred: {e}")
        return {}
    astro_data=output
    return astro_data
     #✅ Step 1: Show Form if No User Details
if st.session_state.user_details is None:
    st.title("Welcome to AstroGPT")
    st.write("Please provide your birth details before starting the chat.")

    with st.form("user_details_form"):
        name = st.text_input("Full Name", placeholder="Enter your name")
        dob = st.date_input("Date of Birth", min_value="1900-01-01", max_value="2025-04-01")
        time_of_birth = st.time_input("Time of Birth")
        place_of_birth = st.text_input("Place of Birth", placeholder="Enter your city")
        submitted = st.form_submit_button("Start Chat")

    if submitted and name and dob and time_of_birth and place_of_birth:
        astro_data = calculate_chart(dob, time_of_birth, place_of_birth)  # Compute Astrology Data
        st.session_state.astro_data = astro_data
        st.session_state.user_details = {
            "name": name,
            "dob": str(dob),
            "time_of_birth": str(time_of_birth),
            "place_of_birth": place_of_birth
        }

        st.session_state.first_message_sent = False
        st.rerun()  # Refresh UI

# ✅ Step 2: Show Chat Interface
if st.session_state.user_details:
    st.title(f"Welcome {st.session_state.user_details['name']} to AstroGPT")
    st.write(
        f"**Birth Details:** {st.session_state.user_details['dob']} | {st.session_state.user_details['time_of_birth']} | {st.session_state.user_details['place_of_birth']}")
    # Display key highlights

    print(st.session_state.astro_data)
    # Display in Streamlit

    # tab1, tab2 = st.tabs(["Chart", "Data"])
    #
    # with tab1:
     # Visual chart
    #
    # with tab2:
    # show_kundli_table(st.session_state.astro_data)  # Table view
    # ✅ Step 3: AI Response to Kundali (Only Once)
    if not st.session_state.first_message_sent:
        user_query = "Tell me about my kundali based on Chart Json" + str(st.session_state.astro_data)
        chart_fig = create_chart(st.session_state.astro_data)
        display_kundli_chart(st.session_state.astro_data)
        response_text = ""

        response_container = st.chat_message("assistant")
        response_placeholder = response_container.empty()


        # ✅ Generate AI response
        for chunk in st.session_state.conversation.run(user_query):
            response_text += chunk
            response_placeholder.write(response_text)
            time.sleep(0.005)

        # ✅ Store response and update flag
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "chart": chart_fig  # Store the figure object
        })
        st.session_state.first_message_sent = True  # ✅ Prevents duplicate first response
        st.rerun()  # Force UI refresh to sync message history

    # ✅ Step 4: Display Chat History (Only Once)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("chart"):
                st.pyplot(message["chart"])
            st.write(message["content"])

    # ✅ Step 5: User Input for Chat
    user_input = st.chat_input("Ask something about your kundali...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        response_placeholder = st.chat_message("assistant").empty()
        response_text = ""

        # ✅ Generate AI response
        for chunk in st.session_state.conversation.run(user_input):
            response_text += chunk
            response_placeholder.write(response_text)
            time.sleep(0.005)

        st.session_state.messages.append({"role": "assistant", "content": response_text})  # Store response