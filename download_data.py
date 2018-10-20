import requests

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, total: x  # If tqdm doesn't exist, replace it with a function that does nothing
    print('**** Could not import tqdm. Please install tqdm for download progressbars! (pip install tqdm) ****')

# Python2 compatibility
try:
    input = raw_input
except NameError:
    pass

download_dict = {
    '1) Kuzushiji-MNIST (10 classes, 28x28, 70k examples)': {
        '1) MNIST data format (ubyte.gz)':
            ['https://storage.googleapis.com/kuzushiji-mnist/train-images-idx3-ubyte.gz',
            'https://storage.googleapis.com/kuzushiji-mnist/train-labels-idx1-ubyte.gz',
            'https://storage.googleapis.com/kuzushiji-mnist/t10k-images-idx3-ubyte.gz',
            'https://storage.googleapis.com/kuzushiji-mnist/t10k-labels-idx1-ubyte.gz'],
        '2) NumPy data format (.npz)':
            ['https://storage.googleapis.com/kuzushiji-mnist/kuzushiji10-train-imgs.npz',
            'https://storage.googleapis.com/kuzushiji-mnist/kuzushiji10-train-labels.npz',
            'https://storage.googleapis.com/kuzushiji-mnist/kuzushiji10-test-imgs.npz',
            'https://storage.googleapis.com/kuzushiji-mnist/kuzushiji10-test-labels.npz'],
    },
    '2) Kuzushiji-49 (10 classes, 28x28, 70k examples)': {
        '1) NumPy data format (.npz)':
            ['https://storage.googleapis.com/kuzushiji-mnist/kuzushiji49-train-imgs.npz',
            'https://storage.googleapis.com/kuzushiji-mnist/kuzushiji49-train-labels.npz',
            'https://storage.googleapis.com/kuzushiji-mnist/kuzushiji49-test-imgs.npz',
            'https://storage.googleapis.com/kuzushiji-mnist/kuzushiji49-test-labels.npz'],
    }
}

# Download a list of files
def download_list(url_list):
    for url in url_list:
        path = url.split('/')[-1]
        r = requests.get(url, stream=True)
        with open(path, 'wb') as f:
            total_length = int(r.headers.get('content-length'))
            print('Downloading {} - {:.1f} MB'.format(path, (total_length / 1024000)))

            for chunk in tqdm(r.iter_content(chunk_size=1024), total=int(total_length / 1024) + 1):
                if chunk:
                    f.write(chunk)
    print('All dataset files downloaded!')

# Ask the user about which path to take down the dict
def traverse_dict(d):
    print('Please select a download option:')
    keys = sorted(d.keys())  # Print download options
    for key in keys:
        print(key)

    userinput = input('> ').strip()

    try:
        selection = int(userinput) - 1
    except ValueError:
        print('Your selection was not valid')
        traverse_dict(d)  # Try again if input was not valid
        return

    selected = keys[selection]

    next_level = d[selected]
    if isinstance(next_level, list):  # If we've hit a list of downloads, download that list
        download_list(next_level)
    else:
        traverse_dict(next_level)     # Otherwise, repeat with the next level

traverse_dict(download_dict)
