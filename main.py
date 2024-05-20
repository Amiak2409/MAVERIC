import pandas as pd
import time
while(True):
    #df = pd.read_json('C:/Users/kondr/source/My project/car_speed.json')
# или
    df = pd.read_json('C:/Users/kondr/source/My project/car_speed.json', orient='index')

    print(df)
    time.sleep(1)