#settings for strikesim

#calendar
start_date = '2025-01-01'
duration = 30
working_days = {'Monday', 'Thursday', 'Friday', 'Wednesday', 'Tuesday'}

#strike pattern settings
strike_pattern = 'indefinite'
weekly_escalation_start = 1

#employer policy
concession_policy = 'none'
retaliation_policy = 'none'
revenue_markup = 1.5
concession_threshold = -10000.0

#union policy
strike_pay_policy = 'fixed'
strike_pay_rate = 0.5
dues_rate = 0.02

#worker parameters
num_workers = 50
initial_wage = 30000.0  # Annual wage (will be converted to daily wage in simulation)
target_wage = 45000.0   # Annual wage (will be converted to daily wage in simulation)
initial_savings_range = (0.0, 5000.0)
initial_morale_range = (0.25, 1.0)
daily_expenditure_rate = 0.95

#financial parameters
initial_employer_balance = 50000.0
initial_strike_fund = 50000.0

#network parameters
lab_size_n = 3
lab_size_prob = 0.5
total_friends_n = 3
total_friends_prob = 0.4
lab_friends_prob = 0.6
department_friends_prob = 0.3
university_friends_prob = 0.1
num_departments = 5
avg_department_size = 20

#union network
bargaining_committee_size = 10
steward_percentage = 0.2
branch_connectedness = 0.7

#network file loading
employer_network_file = 'EmoryEdgeList'
union_network_file = None

#morale parameters
morale_specification = 'sigmoid'
private_morale_alpha = 0.9

#sigmoid morale parameters
inflation = 0.05
belt_tightening = -0.1
sigmoid_gamma = 0.8

#linear morale parameters
linear_alpha = 0.3
linear_beta = 0.3
linear_gamma = 0.4
linear_phi = 0.3

#participation parameters
participation_threshold = 0.5

#policy adjustment parameters
low_participation_threshold = 0.3
high_participation_threshold = 0.8
strike_pay_increase = 0.1
strike_pay_decrease = 0.05
min_strike_pay_rate = 0.3
max_strike_pay_rate = 0.8

#concession parameters
concession_amount = 5.0

#simulation parameters
monte_carlo_simulations = 100
