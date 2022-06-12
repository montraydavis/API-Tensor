import pandas as pd
import flatdict

from endpoint import APIEndpoint
from envs.api_env import APIEnv

def _goal_callback(resp: flatdict):
    try:
        return resp['Success'] == True
    except:
        return False

data_users_columns = ['FIRST_NAME', 'LAST_NAME',
                      'USER_NAME', 'AUTH_PASSWORD']

data_acctype_columns = ['ACC_TYPE_DESC', 'ACC_TYPE_ID']

data_users = [
    ["Montray", "Davis", "@MontrayDavis", 'abc123'],
    ["Montray", "Davis", "@DemoUser1", 'abc456'],
    ["Montray", "Davis", "@DemoUser2", 'abc789']
]

data_account_types = [
    ['Administrator', 0],
    ['Standard User', 1]
]

db = pd.DataFrame(data_users, columns=data_users_columns)
db_acctypes = pd.DataFrame(data_account_types, columns=data_acctype_columns)

db = pd.concat([db, db_acctypes])

endpoint: APIEndpoint = APIEndpoint(
    "http://localhost:8081/login", ["Username", "Password"], "POST", {})

env = APIEnv(endpoint, db, goal_callback=_goal_callback)

env.Learn(n_steps=1000, force_end_success=150, delay=0)
env.Run()