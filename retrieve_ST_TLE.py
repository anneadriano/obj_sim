'''
retrieves TLEs from spacetrack and saves them to TLE file
'''


from spacetrack import SpaceTrackClient

def load_tle(norad_id):

    usr = 'aadriano@uwaterloo.ca'
    pwd = 'AnnronethwAterl00!'
    st = SpaceTrackClient(identity=usr, password=pwd)
    # tle =st.tle_latest(st.tle_latest(norad_cat_id=[norad_id], ordinal=1, format='tle'))

    # Query TLE data for the specified NORAD ID
    query = st.tle_latest(iter_lines=True, norad_cat_id=norad_id, orderby='epoch desc', limit=1)
    
    # Fetch the latest TLE data
    tle_data = list(query)
    
    if tle_data:
        dict = {}
        data = tle_data[0].split(',')
        for item in data:
            # print(item)
            lst = item.lstrip('[{').rstrip('}]').split(':')
            dict[lst[0].strip('"')] = lst[1].strip('"')
        
        return dict

    else:
        return None
    
def save_dict_to_txt(dict, file_path):
    with open(file_path,'w') as file:
        for key, value in dict.items():
            file.write(f"{key}: {value}\n")


'''
IDs:
27424 - AQUA
'''

dict = load_tle(27424)

print(dict)

sat_name = dict['TLE_LINE0']
tle_line1 = dict['TLE_LINE1']
tle_line2 = dict['TLE_LINE2']

print(f"Satellite Name: {sat_name}")
print(tle_line1)
print(tle_line2)

save_dict_to_txt(dict, './TLEs/tle_dict_info.txt')

# Save TLE data to a file
# with open('./TLEs/tle_data_test_AQUA.txt', 'w') as file:
#     file.write(f"{sat_name}\n")
#     file.write(f"{tle_line1}\n")
#     file.write(f"{tle_line2}\n")