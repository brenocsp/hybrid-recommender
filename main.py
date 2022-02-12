import sys
import time

import pandas as pd
import pandas_profiling

def main():
    startTime = time.time()

    with open(sys.argv[1], 'r') as f:
        ratings = pd.read_json(f, lines=True)

    with open(sys.argv[2], 'r') as f:
        content = pd.read_json(f, lines=True)
        
    with open(sys.argv[3], 'r') as f:
        targets = pd.read_csv(f, sep=',', engine='python')

    profile = pandas_profiling.ProfileReport(content)
    profile.to_file('./results/profiler_raw_data.html')
    
    print("Time: %s seconds " % (time.time() - startTime))

main()
