import requests
import os 
import json
from typing import Any, Dict, List
import time
from datetime import datetime
from tqdm import tqdm
import traceback

SINGLE_EVENT_URL = 'https://obserwatoriumbrd.pl/app/api/nodes/post_zdarzenie.php'

def create_form_data_for_request(id):
    return {'zdarzenie_id': id}

def get_ids_list_for_file(filename):
    with open(filename) as f:
        loaded_json = json.load(f)
        accidents_object = loaded_json['mapa']['wojewodztwa'][0]['powiaty'][0]['gminy'][0]

        metadata: Dict[str, Any] = {key: accidents_object[key].strip() if isinstance(accidents_object[key], str) else accidents_object[key] for key in accidents_object if key != 'zdarzenia_detale'}
        
        ids: List[int] = list(map(lambda detail: detail['id'] , accidents_object['zdarzenia_detale']))


    assert len(ids) == metadata['zdarzenia'], f"Corrupted file: expected {metadata['zdarzenia']} accidents, got {len(ids)}"
    return ids, metadata


def scrape_file(filename):
    accidents_ids, metadata = get_ids_list_for_file(filename)

    scraped_details_list = []

    print(f'Scraped file metadata:')
    print(json.dumps(metadata, indent=1))

    tqdm_bar = tqdm(accidents_ids)
    try:
        for id in tqdm_bar:
            tqdm_bar.set_description(str(id))
            try:
                r = requests.post(SINGLE_EVENT_URL, data=create_form_data_for_request(id))
            except Exception:
                error_string = traceback.format_exc()
                tqdm_bar.write(f'Error for id {id}:')
                tqdm_bar.write(error_string)
                continue
            scraped_details_list.append(json.loads(r.text))
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    print(f'Scraped {len(scraped_details_list)} accidents')

    file_name = datetime.now().strftime("%Y_%M_%d_%H_%M_%S") + '_' + os.path.basename(filename)
    metadata_file_name = 'metadata_' + file_name

    abs_file_name = os.path.abspath(file_name)
    abs_metadata_file_name = os.path.abspath(metadata_file_name)

    with open(abs_file_name, 'w') as file:
        json.dump(scraped_details_list, file)
        print(f'Saved scraped data to {abs_file_name}')

    with open(abs_metadata_file_name, 'w') as file:
        json.dump(metadata, file)
        print(f'Saved scraped data to {abs_metadata_file_name}')

    
# scrape_file('src/scraping/all_events_warsaw_2017_to_2018.json')
import glob
files_to_scrape = glob.glob('src/scraping/*.json')
for index, file in enumerate(files_to_scrape):
    print(f'Scraping file {index + 1}/{len(files_to_scrape)}: ' + file)
    scrape_file(file)
