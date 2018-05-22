import json


# iterate through a list of files and return them as json items
def file_based_json_generator(filename_list):
    f_list = filename_list
    if isinstance(filename_list, str):
        f_list = [str]

    for filename in f_list:
        with open(filename) as reader:
            for line in reader:
                value = line.strip()
                if len(value) > 0:
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1].replace('\\"', '"').replace('\\\\"', '\\"')
                    try:
                        json_value = json.loads(value)
                        yield json_value
                    except:
                        pass

    return []
