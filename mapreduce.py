import csv
from collections import defaultdict
import itertools

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
        residence_space = float(row[3])
        building_space = float(row[4])
        exchange_rate = float(row[17])
        unit_price_residence_space = float(row[18])
        unit_price_building_space = float(row[19])

        total_cost = ((unit_price_residence_space * residence_space) +
                      (unit_price_building_space * building_space)) * exchange_rate

        row.append(total_cost)
    return chunk

def reducer(chunks):
    return list(itertools.chain(*chunks))

def split_data(data, num_mappers):
    chunk_size = len(data) // num_mappers
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    return chunks

def map_reduce(header, data, num_mappers, num_reducers):
    # 1. Split the data into chunks for the mappers
    chunks = split_data(data, num_mappers)

    # 2. Map phase: process each chunk in parallel
    mapped_data = [mapper(chunk) for chunk in chunks]

    # 3. Reduce phase: concatenate the chunks back together
    reduced_data = reducer(mapped_data)

    return [header + ['total_cost']] + reduced_data

if __name__ == "__main__":
    # Read the CSV data
    header, data = read_csv("./data/Train_Data.csv")

    # Perform the MapReduce operation
    result = map_reduce(header, data, num_mappers=5, num_reducers=2)

    # Write the result back to a new CSV file
    write_csv("./data/Train_Data_1.csv", result[0], result[1:])
