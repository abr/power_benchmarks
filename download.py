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
print("Downloading benchmark data...")
download_and_unzip(
    gdrive_id='1CubaXJsO-tHGqIZCE_cCJ6HD97hI_5tm',
    filename='benchmark_data.zip', path='./data/')
os.remove('./benchmark_data.zip')  # Delete leftover zip files

# Download pickled data handler
print("Downloading training data...")
download_and_unzip(
    gdrive_id='13JAKf2foL48DH8DO_RD_hx8MCxSdtCcu',
    filename='./abr_keyword_dataset.zip', path='./training/')
os.remove('./abr_keyword_dataset.zip')  # Delete leftover zip files

# Download 180 neuron trained weights
print("Downloading trained weights...")
download_and_unzip(
    gdrive_id='1VQADy5sk46KjE1gu86RILJ2yhv2vovL-',
    filename='./trained_weights_180.pkl.zip', path='./data/')
os.remove('./trained_weights_180.pkl.zip')  # Delete leftover zip files

# Download audio clips
print("Downloading audio clips...")
os.system('mkdir -p ./demo/audio')
download_and_unzip(
    gdrive_id='1R9EncCTMIWFXut-3UVXX3s9QVdOL9P7y',
    filename='./audio_clips_for_demo.zip', path='./demo/audio/')
os.remove('./audio_clips_for_demo.zip')  # Delete leftover zip files
