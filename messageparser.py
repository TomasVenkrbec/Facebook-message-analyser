from __future__ import unicode_literals
import argparse
import os
import json
import re
import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import dateutil.parser
from tzlocal import get_localzone
from collections import Counter
from datetime import datetime

ORD_EMOJI_CUTOFF = 1000 # Smallest decimal unicode value of characters that are considered being an emoji
NON_EMOJI_ORDS = [65039, 8205, 8211, 8222, 8220, 8217, 8212, 1618, 127995] # Special unicode characters that are not emojis, but appear near emojis often (variation selectors, etc.)
AVERAGE_MESSAGE_ROLLING_WINDOW = 30 # How big (in days) the rolling window for average messages over time is
TOP_EMOJI_COUNT = 20 # How many most used emoji are shown in graph

class Message:
    def __init__(self, time, message_type, content, author, platform):
        self.message_type = message_type
        self.content = content
        self.author = author
        self.time = time
        self.platform = platform


class Conversation:
    def __init__(self, path, participants_fb, participants_discord):
        self.participants_fb = participants_fb
        self.participants_discord = participants_discord
        self.path = path
        self.messages = []
        self.weekday_frequency = {}
        self.hour_frequency = {}
        self.day_frequency = {}
        self.emoji_frequency = Counter({})
        self.participants_message_count = {}
        self.participants_message_length = {participant:[] for participant in (self.participants_fb + self.participants_discord)} # Prepare dict entry for individual participants

    def add_message(self, message):
        self.messages.append(message)

    def get_message_count(self):
        print(f"Total Facebook message count: {len([msg for msg in self.messages if msg.platform == 'facebook'])}")
        print(f"Total Discord message count: {len([msg for msg in self.messages if msg.platform == 'discord'])}")
        print(f"Total message count: {len(self.messages)}")

    def get_weekday_frequency(self):
        for message in self.messages:
            weekday = message.time.split(" ")[0] # Get the day from date string
            if weekday not in self.weekday_frequency:
                self.weekday_frequency[weekday] = 0
            self.weekday_frequency[weekday] += 1

    def get_hour_frequency(self):
        for message in self.messages:
            hour = message.time.split(":")[0].split(" ")[-1] # Get the hour number from date string (splitting based on : because there may be inconsistency in number of spaces)
            if hour not in self.hour_frequency:
                self.hour_frequency[hour] = 0
            self.hour_frequency[hour] += 1

    def get_day_frequency(self):
        for message in self.messages:
            day = f"{message.time.split()[1]} {message.time.split(':')[0].split()[-2]} {message.time.split()[-1]}"
            if day not in self.day_frequency:
                self.day_frequency[day] = 0
            self.day_frequency[day] += 1
    
    def get_emoji_frequency(self):
        for message in self.messages:
            emojis = re.findall(r'[^\w\s,]', message.content)
            for emoji in emojis:
                if ord(emoji) > ORD_EMOJI_CUTOFF and ord(emoji) not in NON_EMOJI_ORDS:
                    if emoji not in self.emoji_frequency:
                        self.emoji_frequency[emoji] = 0
                    self.emoji_frequency[emoji] += 1

    def analyze_message_length(self):
        for message in self.messages:
            if message.author not in self.participants_message_count:
                self.participants_message_count[message.author] = 0
            self.participants_message_count[message.author] += 1
            self.participants_message_length[message.author].append(len(message.content))
    
    def create_graphs(self):
        if self.weekday_frequency:
            plt.title("Message frequency during weekdays")
            plt.xlabel("Days of week")
            plt.ylabel("Message count")
            x, y = zip(*sorted(self.weekday_frequency.items()))

            # Sort by days of week
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            indices = []
            for day in days:
                indices.append(x.index(day))
            x = np.array(x)[indices]
            y = np.array(y)[indices]

            plt.bar(x, y)
            plt.show()
        
        if self.hour_frequency:
            plt.title("Message frequency during hours of day")
            plt.xlabel("Hours of day")
            plt.ylabel("Message count")
            x, y = zip(*sorted(self.hour_frequency.items()))
            plt.bar(x, y)
            plt.show()

        if self.day_frequency:
            plt.title("Message frequency over time")
            plt.xlabel("Day")
            plt.ylabel("Message count")
            
            # Sort by date and divide into x and y axis
            x = sorted(self.day_frequency.items(), key=sortkey_date)
            x, y = zip(*x)

            # Make 7-day rolling average using discrete convolution
            y = np.convolve(y, np.ones(AVERAGE_MESSAGE_ROLLING_WINDOW), 'valid') / AVERAGE_MESSAGE_ROLLING_WINDOW
            x = x[AVERAGE_MESSAGE_ROLLING_WINDOW-1:] # Cut off starting dates so the lengths are the same (convolution without padding cuts edges)
            x_positions, x_ticks = calculate_ticks_month_year([y.split()[0] for y in x], [y.split()[2] for y in x]) # Get months and years
            plt.xticks(x_positions, x_ticks, rotation=45)

            plt.plot(x, y)
            plt.show()
        
        if self.participants_message_count:
            plt.title("Message count from conversation participants")
            plt.xlabel("Participant")
            plt.ylabel("Message count")
            x, y = zip(*sorted(self.participants_message_count.items()))
            plt.bar(x, y)
            plt.show()
        
            x = np.linspace(0, 200)
            plt.title("Message length probability from conversation participants")
            plt.xlabel("Message length")
            plt.ylabel("Probability density")

            participant_legend = []
            for participant in self.participants_fb + self.participants_discord:
                participant_legend.append(participant)
                mean = np.mean(self.participants_message_length[participant])
                std = np.std(self.participants_message_length[participant])
                plt.plot(x, scipy.stats.norm(loc=mean, scale=std).pdf(x))

            plt.legend(participant_legend)
            plt.show()
        
        if self.emoji_frequency:
            plt.title("Emoji usage frequency in conversation")
            plt.xlabel("Emoji")
            plt.ylabel("Emoji usage count")

            x, y = zip(*self.emoji_frequency.most_common(TOP_EMOJI_COUNT))
            plt.xticks([])
            plt.bar(x, y)
            print(f"Since matplotlib doesn't support emojis, here's the emoji list:\n{''.join(str(emoji + ' ') for emoji in x)}")
            plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="Parse Facebook messages from downloaded JSON data dump.")
    parser.add_argument("--path", dest="path", action="store", required=True, help="Name of folder with individual JSON files.")
    return parser.parse_args()

def calculate_ticks_month_year(months, years):
    positions = []
    ticks = []
    full_dates = [f"{x} {y}" for x,y in zip(months, years)]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    years = np.unique(np.array(years))
    for year in years:
        for month in months:
            full_date = f"{month} {year}"
            if full_date in full_dates:
                positions.append(full_dates.index(full_date))
                ticks.append(full_date)

    return positions, ticks

def sortkey_date(x):
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return [x[0].split()[2], months.index(x[0].split()[0]), "{0:0>2}".format(x[0].split()[1])]    

def verify_path(path):
    if not os.path.exists(path):
        print("Error: Selected folder with messages doesn't exist.")
        exit(-1)
    if not os.path.isfile(path + "/message_1.json"):
        print("Error: Selected folder doesn't contain message files or message_1.json is missing.")
        exit(-1)

def load_conversation_metadata(path):
    with open(path + "/message_1.json", "r") as first_file:
        messages = json.load(first_file)

    participants_fb = []
    print("Participants on Facebook:")
    for user in messages["participants"]:
        print(user["name"].encode("latin1").decode("utf8"))
        participants_fb.append(user["name"].encode("latin1").decode("utf8"))

    last_message_timestamp = messages["messages"][0]["timestamp_ms"]
    print(f"Last Facebook message sent at: {datetime.fromtimestamp(last_message_timestamp/1000).ctime()}\n")

    with open(glob.glob("kiki/[!message_*.json]*")[0], "r") as first_file: # Load first file with non-facebook format
        messages = json.load(first_file)

    participants_discord = set()
    print("Participants on Discord:")
    for msg in messages["messages"]:
        participants_discord.add(msg["author"]["nickname"].encode("utf-16", 'surrogatepass').decode("utf-16"))
    participants_discord = list(participants_discord)
    for participant in participants_discord:
        print(participant)
    last_message_timestamp = messages["messages"][-1]["timestamp"]
    print(f"Last Discord message sent at: {dateutil.parser.parse(last_message_timestamp).astimezone(get_localzone()).ctime()}\n")

    conversation = Conversation(path, participants_fb, participants_discord)
    return conversation

def convert_symbols_to_emojis(message):
    convertable_emojis = {":)":"🙂", ":D":"😀", "^_^":"😊", "O:)":"😇", ":*":"😗", ":P":"😛", "8)":"😎", "(y)":"👍", 
                        ":(":"😞", ":/":"😕", ":'(":"😢", "o.O":"😳", "O.o":"😳", "-_-":"😑", ":|":"😐", "<3":"❤️"}

    message = ' '.join([convertable_emojis.get(i, i) for i in message.split()])

    return message

def load_messages(conversation):
    file_list = glob.glob(f"{conversation.path}/*.json")
    file_list.sort(key=lambda x: "{0:0>20}".format(x).lower()) # Workaround for natural string sort to work
    for file_name in file_list:
        with open(file_name, "r") as file:
            messages = json.load(file)

            if "channel" in messages:
                used_app = "discord"
            elif "participants" in messages:
                used_app = "facebook"

            for message in messages["messages"]:
                message_content = ""
                if "content" in message:
                    message_type = "text"
                    if used_app == "facebook":
                        message_content = message["content"].encode("latin1").decode("utf8")
                    elif used_app == "discord":
                        message_content = message["content"].encode("utf-16", 'surrogatepass').decode("utf-16")
                    message_content = convert_symbols_to_emojis(message_content)
                elif "videos" in message:
                    message_type = "video"
                elif "photos" in message:
                    message_type = "photo"
                elif "sticker" in message:
                    message_type = "sticker"
                elif "gifs" in message:
                    message_type = "gif"
                elif "files" in message:
                    message_type = "file"
                elif "audio_files" in message:
                    message_type = "audio"
                elif message["is_unsent"]:
                    message_type = "deleted"
                else:
                    continue # Message was not properly saved by Facebook

                if used_app == "facebook":
                    message_date = message["timestamp_ms"]
                    message_date = datetime.fromtimestamp(message_date/1000).ctime()
                    message_author = message["sender_name"].encode("latin1").decode("utf8")
                elif used_app == "discord":
                    message_date = message["timestamp"]
                    message_date = datetime.fromisoformat(message_date.split(".")[0]).astimezone(get_localzone()).ctime()
                    message_author = message["author"]["name"]

                conversation.add_message(Message(message_date, message_type, message_content, message_author, used_app))

def main():
    # Validate files
    args = parse_args()
    verify_path(args.path)

    # Load files
    conversation = load_conversation_metadata(args.path)
    load_messages(conversation)

    # Analyze files
    conversation.get_message_count()
    conversation.get_weekday_frequency()
    conversation.get_hour_frequency()
    conversation.get_emoji_frequency()
    conversation.get_day_frequency()
    conversation.analyze_message_length()

    # Create graphs
    conversation.create_graphs()

if __name__ == "__main__":
    main()