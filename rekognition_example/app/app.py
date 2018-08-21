import boto3
import requests
import glob

import asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db import File, Label, Base


def download():
    session = boto3.Session(profile_name='default')
    rekognition = session.client('rekognition')
    response = requests.get('https://cdn-images-1.medium.com/max/800/1*sB1y4OTS4VCNU9p2gxxkdg.png')
    response_content = response.content

    rekognition_response = rekognition.detect_faces(Image={'Bytes': response_content}, Attributes=['ALL'])
    # import pudb
    # pudb.set_trace()
    print(rekognition_response)

import sys

if len(sys.argv) < 2:
    print("We add an argument")
    exit()


async def run(path_image, session, session_maker_db):
    with open(path_image, 'rb') as image:
        session_db = session_maker_db()
        client = session.client('rekognition')
        response = client.detect_labels(Image={'Bytes': image.read()})
        print('Detected labels in ' + path_image)
        fil = File(filepath=path_image)
        session_db.add(fil)
        session_db.commit()
        for label in response['Labels']:
            name = label['Name']
            confidence = label['Confidence']
            print(label['Name'] + ' : ' + str(label['Confidence']))
            label = Label(name=name, confidence=confidence, fil=fil)
            session_db.add(label)
            session_db.commit()


path_images = sys.argv[1]
list_path_images = glob.iglob(path_images)
session_boto3 = boto3.Session(profile_name='default')
engine = create_engine('sqlite:///database.db')
Base.metadata.bind = engine
session_maker_db = sessionmaker(bind=engine)

loop = asyncio.get_event_loop()

list_requests = []
for path_image in list_path_images:
    print("path_image: " + path_image)
    list_requests.append(asyncio.ensure_future(run(path_image, session_boto3, session_maker_db)))

# responses = loop.run_until_complete(asyncio.gather(*list_requests))
loop.run_until_complete(asyncio.gather(*list_requests))

loop.close()
