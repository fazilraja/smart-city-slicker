from project3 import load_data

def test_load_data():
    file = 'smartcity/AK Anchorage.txt'
    data = load_data(file)
    print(data[3])
    assert 0 == 0