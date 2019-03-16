import csv


def insert_csv_header(path):
    with open(path, newline='') as f:
        r = csv.reader(f)
        data = [line for line in r]

    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['date', 'time', 'open', 'high', 'low', 'close', 'volume'])
        w.writerows(data)