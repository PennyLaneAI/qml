import time
import numpy as np
import pandas as pd

import click

@click.command()
@click.option('-c', is_flag=True, help='Pickup from where you left off.')
def cli(c):

    with open("default_qubit_demos.log", 'r') as f:
        default_qubit_demos = f.readlines()

    starting_index = 0 
    num_demos = len(default_qubit_demos)
    default_qubit_demos = np.array([default_qubit_demos[i].replace(".py\n", "") for i in range(num_demos)])

    if c:
        df = pd.read_csv('lightning_versus_default.csv', header=None)
        last_demo = df.iloc[-1:][1].item()
        print("Last demo tested:", last_demo)
        index = default_qubit_demos.tolist().index(last_demo)
        starting_index = index + 1
        default_qubit_demos = default_qubit_demos[starting_index:]

    lightning_qubit_demos = np.array(["lightning_" + default_qubit_demos[i].replace(".py\n", "") for i in range(num_demos - starting_index)])
    for i, (default_demo, lightning_demo) in enumerate(zip(default_qubit_demos, lightning_qubit_demos)):

        print()
        print("##############################################")
        print("Running", default_demo) 
        print("##############################################")
        print()
        
        runtimes = np.array([-99, -99], dtype='float')  
        error_msg = ""

        try: 
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("Default qubit") 
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            start = time.time()
            exec(f'import {default_demo}')
            total = time.time() - start

            runtimes[0] = total
            
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("Lightning qubit") 
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            start = time.time()
            exec(f'import {lightning_demo}')
            total = time.time() - start

            runtimes[1] = total

        except Exception as e:
            error_msg += str(e)
            
        df = pd.DataFrame({"Tutorial": default_demo, "Default runtime": runtimes[0], "Lightning runtime": runtimes[1], "errors": error_msg}, index=[i + starting_index])
        df.to_csv("lightning_versus_default.csv", mode='a', header=False)

if __name__ == '__main__':
    cli()