from torch_seed_rl import dummy

def test_capital_case():
    assert dummy.capital_case('semaphore') == 'Semaphore'