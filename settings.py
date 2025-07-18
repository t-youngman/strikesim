#settings for strikesim

#calendar
start_date = '2025-01-01'
duration = 100 #in days
working_days = {'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'}

#strike pattern settings
strike_pattern = 'indefinite'  # 'indefinite', 'once_a_week', 'once_per_month', 'weekly_escalation'
weekly_escalation_start = 1  # Starting number of days for weekly escalation

#employer policy
#concession_policy = 'none'
#retaliation_policy = 'none'
#revenue_markup = 1.5
#concession_threshold = -10000.0

#union policy
strike_pay_policy = 'fixed'
strike_pay_rate = 0.5  # As fraction of normal wage
dues_rate = 0.05  # As fraction of wage

#worker parameters
num_workers = 50
initial_wage = 100.0
target_wage = 105.0  # Higher target wage for more motivation
initial_savings_range = (500.0, 2000.0)  # Higher initial savings
initial_morale_range = (0.0, 1.0)  # Higher initial morale
daily_expenditure_rate = 0.95  # Daily expenses as fraction of daily wage (1.0 = full daily wage)

#financial parameters
initial_employer_balance = 50000.0  # Lower employer balance
initial_strike_fund = 100000.0  # Higher strike fund

#network parameters
#employer network (university-style)
lab_size_n = 3  # Negative binomial parameter n for lab sizes
lab_size_prob = 0.50  # Negative binomial parameter p for lab sizes
total_friends_n = 3  # Negative binomial parameter n for total friends
total_friends_prob = 0.4  # Negative binomial parameter p for total friends
lab_friends_prob = 0.6  # Probability of connections to lab mates
department_friends_prob = 0.3  # Probability of connections to department mates
university_friends_prob = 0.1  # Probability of connections to university mates
num_departments = 5  # Number of departments to generate (if synthetic data needed)
avg_department_size = 20  # Average department size (if synthetic data needed)

#union network
bargaining_committee_size = 3
steward_percentage = 0.1  # 10% of workers are union stewards
branch_connectedness = 0.7  # Higher density

#network file loading (set to None to use generated networks, or specify filename without .gexf extension)
employer_network_file = None  # e.g., 'defra' to load networks/employers/defra.gexf
union_network_file = None     # e.g., 'union_network' to load networks/unions/union_network.gexf

#morale parameters
morale_specification = 'sigmoid'  # 'sigmoid', 'linear'
social_morale_beta = 0.5  # More weight on social factors

#sigmoid morale parameters
inflation = 0.05
belt_tightening = -0.2
sigmoid_gamma = 0.8

#linear morale parameters
linear_alpha = 0.3
linear_beta = 0.3
linear_gamma = 0.4
linear_phi = 0.3

#participation parameters
participation_threshold = 0.5  # Lower threshold

#policy adjustment parameters
low_participation_threshold = 0.3
high_participation_threshold = 0.8
strike_pay_increase = 0.1
strike_pay_decrease = 0.05
min_strike_pay_rate = 0.3
max_strike_pay_rate = 0.8

#concession parameters
#concession_amount = 5.0

#simulation parameters
monte_carlo_simulations = 100