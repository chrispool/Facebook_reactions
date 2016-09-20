"""
A simple example script to get all posts on a user's timeline.
Originally created by Mitchell Stewart.
<https://gist.github.com/mylsb/10294040>
"""
import facebook
import requests
from collections import Counter, defaultdict
import json

APP_ID = '794145884053786'
SECRET_KEY = 'b2e7edda34a2a09388c8c855eac502a7'
graph = facebook.GraphAPI(access_token='{}|{}'.format(
    APP_ID, SECRET_KEY), version='2.6')


#allready done: 'universityofgroningen', '200273583406054',FoxNews, cnn 
pages = ['Disney', 'FoxNews', 'cnn', 'ESPN', 'HuffPostWeirdNews', 'nickelodeon', 'nytimes', 'theguardian', 'time', 'DailyMail']


def emotion_vector(counter_object):
    """NONE, LIKE, LOVE, WOW, HAHA, SAD, ANGRY, THANKFUL"""
    
    vector = [counter_object[emotion] for emotion in possible_reactions]
    return vector

mapping = {
    'fairy_tales' : {2 : 'ANGER', 3: 'NONE', 4: 'JOY', 5: 'NONE', 6: 'SADNESS', 7: 'SUPRISE'},
    'ISEAR': {'anger': 'ANGER', 'disgust': 'NONE', 'FEAR' : 'NONE', 'joy': 'JOY', 'sadness': 'SADNESS', 'shame': 'NONE', 'guilt': 'NONE'},
    'sem_eval'    : {'anger' : 'ANGER', 'disgust': 'NONE', 'fear': 'None', 'joy': 'JOY', 'sadness': 'SADNESS', 'suprise': 'SUPRISE'},
}


def get_all(items):
    """Wrap this block in a while loop so we can keep paginating requests until finished."""
    all_items = []
    i = 0
    print("Getting data", end="")
    while len(all_items) < 3000:
        print("*", end="")
        try:
            # Perform some action on each post in the collection we receive from
            # Facebook.
            all_items.extend(items['data'])
            # Attempt to make a request to the next page of data, if it exists.
            items = requests.get(items['paging']['next']).json()
        except KeyError:
            # When there are no more pages (['paging']['next']), break from the
            # loop and end the script.
            print("....Finished getting data")
            break
        
    return all_items


possible_reactions = ['NONE', 'LIKE', 'LOVE', 'HAHA', 'WOW', 'SAD', 'ANGRY', 'THANKFUL']
result = []
for page_id in pages:
    
    all_posts = get_all(graph.get_object(id="{}/feed".format(page_id)))
    for i, post in enumerate(all_posts):
        print("Processing post {} of {}".format(i, len(all_posts)))
        if 'message' in post:
            
            # reaction_vector = []
            # for reaction in possible_reactions:
            #     rs = graph.get_object(id="{}?fields=reactions.type({}).summary(true)".format(post['id'], reaction))
            #     reaction_vector.append(rs['reactions']['summary']['total_count'])
            # print(possible_reactions)
            # print(reaction_vector)
                
            result.append([{'created_time': post['created_time'], 'message': post[
                          'message'], 'reactions': []}])

    with open('data/complete.json', 'w') as outfile:
        json.dump(result, outfile, sort_keys = True, indent = 4, ensure_ascii=False)
    print("Saved json file")
