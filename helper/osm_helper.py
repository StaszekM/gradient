import overpy

def get_all_values_for_tag(tag: str):
    """
    Gets all unique values for given OSM key.
    Helpful for semi-automatic OSM ke:walue pair retrieval.
    Returns list of unique values for given tag.
    """
    api = overpy.Overpass()
    query = f"""
    area[name="Wrocław"];
    (nwr[{tag}](area););
    out tags;
    """ 
    values_all = []
    result = api.query(query)
    for way in result.ways:
        for key, value in way.tags.items():
            if key ==tag:
                print(f"Way {way.id}: {key} = {value}")
                values_all.append(value)

    for node in result.nodes:
        for key, value in node.tags.items():
            if key ==tag:
                print(f"Node {node.id}: {key} = {value}")
                values_all.append(value)

    for relation in result.relations:
        for key, value in relation.tags.items():
            if key ==tag:
                print(f"Relation {relation.id}: {key} = {value}")
                values_all.append(value)
    values_set = set(values_all)
    return list(values_set)


def slice_list(input_list, max_elements_per_slice=20):
    """
    Helper function for list slicing - useful when loading large amount of data using OSMOnineloader
    Returns sliced list
    """
    sliced_list = []
    for i in range(0, len(input_list), max_elements_per_slice):
        sliced_list.append(input_list[i:i + max_elements_per_slice])
    return sliced_list