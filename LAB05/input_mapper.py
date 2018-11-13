from pandas import DataFrame


def mapper(x):
    if x == 'duzy':
        return 2
    if x == 'sredni':
        return 1
    if x == 'tak':
        return 1
    if x == 'nie':
        return 0
    else:
        return x


def map_input(input_data: DataFrame):
    for column in input_data:
        input_data[column] = input_data[column].map(mapper)
    return input_data
