import csv
import itertools
from collections import defaultdict
import pandas as pd

def read_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        data = [row for row in reader]
    return header, data

def write_csv(file_path, header, data):
    with open(file_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)

def mapper(chunk):
    for row in chunk:
        key = row[0]
        value = [row[4], row[5], row[18], row[19], row[20]]
        yield (key, value)

def reducer(key, values):
    total_costs = []
    for value in values:
        residence_space = float(value[0])
        building_space = float(value[1])
        exchange_rate = float(value[2])
        unit_price_residence_space = float(value[3])
        unit_price_building_space = float(value[4])

        total_cost = ((unit_price_residence_space * residence_space) +
                      (unit_price_building_space * building_space)) * exchange_rate
        total_costs.append([key] + value + [total_cost])
    return total_costs

def split_data(data, num_mappers):
    chunk_size = len(data) // num_mappers
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    return chunks

def index_data(data):
    return [[str(i)] + row for i, row in enumerate(data)]

def map_reduce(header, data, num_mappers, num_reducers):
    # Index the data
    data = index_data(data)

    # 1. Split the data into chunks for the mappers
    chunks = split_data(data, num_mappers)

    # 2. Map phase: process each chunk in parallel
    mapped_data = list(itertools.chain(*[list(mapper(chunk)) for chunk in chunks]))

    # 3. Group by key
    grouped_data = defaultdict(list)
    for key, value in mapped_data:
        grouped_data[key].append(value)

    # 4. Reduce phase: calculate the total cost for each row
    reduced_data = [reducer(key, values) for key, values in grouped_data.items()]

    return [['index'] + header[3:5] + header[17:20] + ['total_cost']] + list(itertools.chain(*reduced_data))

if __name__ == "__main__":
    # Read the CSV data
    header, data = read_csv("/workspaces/Big-data-course-project/data/Train_Data.csv")

    # Perform the MapReduce operation
    result = map_reduce(header, data, num_mappers=5, num_reducers=2)

    # Write the result back to a new CSV file
    write_csv("/workspaces/Big-data-course-project/data/output.csv", result[0], result[1:])

    df = pd.read_csv('/workspaces/Big-data-course-project/data/output.csv', index_col=None)

    df.info()
    print(df.iloc[0])
