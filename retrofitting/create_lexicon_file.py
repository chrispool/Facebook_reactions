from collections import defaultdict

filename = 'lexicons/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt'
print("start")
lexicon = defaultdict(list)
with open(filename, 'r') as f:
    lines = f.readlines()
for i in range(0,len(lines), 10): # each word had 10 lines
    for line in lines[i:i+10]:
        cols = line.strip().split()
        if cols[2] == "1":
            lexicon[cols[1]].append(cols[0])


print(lexicon.keys())
# >>> dict_keys(['trust', 'joy', 'anger', 'sadness', 'surprise', 'positive', 'disgust', 'anticipation', 'negative', 'fear'])
emotion_to_ignore = []

lexicon_file = open('lexicons/NRC.txt', 'w')
#create a line for each emotion, and each word
for emotion in lexicon:
    if emotion not in emotion_to_ignore:
        for word in lexicon[emotion]:
            remaining = [w for w in lexicon[emotion] if w != word]
            entry = [word] + remaining
            lexicon_file.write("{}\n".format(" ".join(entry)))

lexicon_file.close()

   

