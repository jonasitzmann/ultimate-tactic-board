import os
from pathlib import Path
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors

client_secrets_file = "client_secrets.json"
scopes = ["https://www.googleapis.com/auth/youtube.upload"]


def upload_video(file: Path, title, thumbnail):
    # Get the OAuth 2.0 credentials
    flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
        client_secrets_file, scopes
    )
    credentials = flow.run_local_server(port=8080)
    youtube = googleapiclient.discovery.build("youtube", "v3", credentials=credentials)
    request = youtube.videos().insert(
        part="snippet,status",
        body={
            "snippet": {
                "categoryId": "22",  # Example: People & Blogs
                "description": "",
                "title": title,
            },
            "status": {"privacyStatus": "unlisted"},  # "private", "public", "unlisted"
        },
        media_body=str(file.absolute()),
    )
    response = request.execute()
    print(response)

    # Upload the thumbnail
    if thumbnail:
        youtube.thumbnails().set(
            videoId=response["id"],
            media_body=googleapiclient.http.MediaFileUpload(thumbnail),
        ).execute()

    return response
