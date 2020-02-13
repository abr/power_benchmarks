import os
import shutil
import requests
import zipfile


# Function adapted from SO for dealing with GDrive download weirdness
def download_and_unzip(gdrive_id, filename, path):

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keepalive new chunks
                   f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': gdrive_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': gdrive_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, filename)

    with zipfile.ZipFile(filename) as zfile:
        for item in zfile.namelist():
            filename = os.path.basename(item)
            # skip directories
            if not filename:
                continue

            source = zfile.open(item)
            target = open(os.path.join(path, filename), 'wb')
            with source, target:
                shutil.copyfileobj(source, target)


# Download pickled data handler
download_and_unzip(
    gdrive_id='1CubaXJsO-tHGqIZCE_cCJ6HD97hI_5tm',
    filename='benchmark_data.zip', path='./data/')

# Delete leftover zip files
os.remove('./benchmark_data.zip')

# Download pickled data handler
download_and_unzip(
    gdrive_id='13JAKf2foL48DH8DO_RD_hx8MCxSdtCcu',
    filename='./abr_keyword_dataset.zip', path='./training/')

# Delete leftover zip files
os.remove('./abr_keyword_dataset.zip')

# Download 180 neuron trained weights
download_and_unzip(
    gdrive_id='16K2k9SoDUVI8hviVxcYGHiJf9y2jWTex',
    filename='./trained_weights_180.pkl.zip', path='./data/')

# Delete leftover zip files
os.remove('./trained_weights_180.pkl.zip')

# Download audio clips
download_and_unzip(
    gdrive_id='16K2k9SoDUVI8hviVxcYGHiJf9y2jWTex',
    filename='./audio_clips_for_demo.zip', path='./demo/audio')

# Delete leftover zip files
os.remove('./audio_clips_for_demo.zip')
