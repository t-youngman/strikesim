#settings for strikesim

#calendar
start_date = '2025-01-01'
duration = 100 #in days
working_days = {'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'}

#employer policy
concession_policy = 'none'
retaliation_policy = 'none'
revenue_markup = 1.5
concession_threshold = -10000.0

#union policy
strike_pay_policy = 'fixed'
strike_pay_rate = 0.5  # As fraction of normal wage
dues_rate = 0.02  # As fraction of wage

#worker parameters
num_workers = 50
initial_wage = 100.0
target_wage = 150.0  # Higher target wage for more motivation
initial_savings_range = (500.0, 2000.0)  # Higher initial savings
initial_morale_range = (0.6, 0.9)  # Higher initial morale
initial_strike_participation_rate = 0.7  # Higher initial participation

#financial parameters
initial_employer_balance = 50000.0  # Lower employer balance
initial_strike_fund = 100000.0  # Higher strike fund

#network parameters
#employer network
executive_size = 3
department_size = 5
team_size = 8

#union network
bargaining_committee_size = 3
department_density = 0.5  # Higher density
team_density = 0.7  # Higher density

#morale parameters
morale_specification = 'sigmoid'  # 'sigmoid', 'linear', 'no_motivation'
private_morale_alpha = 0.6
social_morale_beta = 0.4  # More weight on social factors

#sigmoid morale parameters
sigmoid_alpha = 1.0
sigmoid_beta = 1.0
sigmoid_gamma = 0.5

#linear morale parameters
linear_alpha = 0.3
linear_beta = 0.3
linear_gamma = 0.4

#no_motivation morale parameters
no_motivation_alpha = 0.5
no_motivation_beta = 0.3
no_motivation_gamma = 0.2

#participation parameters
participation_threshold = 0.4  # Lower threshold

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

