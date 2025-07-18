import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# --- City Population Data (unchanged) ---
city_populations = {
    "New York": 8_468_000, "Los Angeles": 3_849_000, "Chicago": 2_697_000,
    "Houston": 2_303_000, "Phoenix": 1_644_000, "Philadelphia": 1_576_000,
    "San Antonio": 1_472_000, "San Diego": 1_381_000, "Dallas": 1_289_000,
    "Austin": 974_000, "Jacksonville": 972_000, "Fort Worth": 956_000,
    "San Francisco": 808_000, "Columbus": 907_000, "Charlotte": 911_000,
    "Indianapolis": 880_000, "Seattle": 733_000, "Denver": 711_000,
    "Washington": 672_000, "Boston": 651_000
}
total_population = sum(city_populations.values())
city_weights_all = {city: pop / total_population for city, pop in city_populations.items()}
cities_all = list(city_weights_all.keys())
weights_all = list(city_weights_all.values())


# --- Entity Pools (unchanged) ---
entity_pools = {
    "Credit Card (POS/Online)": {
        "entity_types": ["Grocery", "Retail", "Restaurant", "Gas Station", "Online Service", "Travel", "Healthcare"],
        "specific_entities": {
            "Grocery": ["SuperMart", "FreshFoods", "CornerMarket"],
            "Retail": ["FashionHub", "TechGadgets", "BookWorm", "HomeGoods"],
            "Restaurant": ["PizzaPalace", "SushiSpot", "CafeConnect", "BurgerJoint"],
            "Gas Station": ["SpeedyGas", "FuelStop"],
            "Online Service": ["StreamFlix", "CloudStoragePro", "FitnessApp"],
            "Travel": ["AirLink", "RoadTrips"],
            "Healthcare": ["MediCare Pharmacy", "Dr. Visit Co-pay"]
        }
    },
    "Online Purchase (Non-Card)": {
        "entity_types": ["E-commerce Giant", "Specialty Store", "Subscription Service"],
        "specific_entities": {
            "E-commerce Giant": ["Amazon", "Walmart.com", "Target.com", "eBay"],
            "Specialty Store": ["GadgetBay", "FashionNovaOnline", "BookWorld.com"],
            "Subscription Service": ["Netflix", "Spotify", "Hulu", "Adobe Creative Cloud"]
        }
    },
    "Venmo/PayPal Transfer": {
        "entity_types": ["Person-to-Person", "Small Business"],
        "specific_entities": {
            "Person-to-Person": [f"1{i:09d}" for i in range(100, 200)],
            "Small Business": ["Local Coffee Shop", "Freelance Designer", "Yoga Studio"]
        }
    },
    "Bank Transfer (ACH)": {
        "entity_types": ["Utility Bill", "Rent Payment", "Loan Payment", "Friend/Family Transfer", "Investment"],
        "specific_entities": {
            "Utility Bill": ["PowerCo", "WaterWorks", "City Gas"],
            "Rent Payment": ["Landlord Management LLC"],
            "Loan Payment": ["StudentLoanProvider", "AutoFinanceCo"],
            "Friend/Family Transfer": [f"7{i:07d}" for i in range(50, 70)],
            "Investment": ["InvestGrow Securities"]
        }
    }
}


def generate_synthetic_transactions(
    start_date: datetime,
    end_date: datetime,
    avg_daily_transactions: int = 10,
    accident_probability: float = 0.03, # Back-to-back transactions
    max_accident_transactions: int = 3,
    high_amount_accident_probability: float = 0.005, # New: Chance of unusually high amount
    high_amount_multiplier: float = 3.0, # New: Amount will be this many times max_amt
    mistyped_entity_accident_probability: float = 0.003, # New: Chance of entity typo
    customer_id: str = "customer_001",
    home_city: str = None,
    home_city_bias_probability: float = 0.85,
    cities_list_all: list = cities_all,
    city_selection_weights_all: list = weights_all,
    entity_pools_config: dict = entity_pools
) -> pd.DataFrame:
    """
    Generates synthetic transaction data for a single customer, incorporating realistic
    time-based patterns, various "accident" scenarios (back-to-back, high-amount, mistyped entity),
    population-weighted city locations (with a home city bias), and transaction-specific entities.
    """

    if home_city is None:
        home_city = random.choices(cities_list_all, weights=city_selection_weights_all, k=1)[0]
    
    other_cities = [city for city in cities_list_all if city != home_city]
    if not other_cities:
        other_city_weights = []
    else:
        other_city_weights_raw = [city_weights_all[city] for city in other_cities]
        total_other_population = sum(other_city_weights_raw)
        other_city_weights = [w / total_other_population for w in other_city_weights_raw]


    transactions = []
    current_time = start_date

    hourly_frequency_multipliers = {
        0: 0.01, 1: 0.01, 2: 0.01, 3: 0.01, 4: 0.01, 5: 0.02, 6: 0.05,
        7: 0.2, 8: 0.5, 9: 0.8, 10: 1.0, 11: 1.2, 12: 1.5, 13: 1.8,
        14: 1.2, 15: 1.0, 16: 0.9, 17: 1.1, 18: 1.0, 19: 0.8, 20: 0.7,
        21: 0.5, 22: 0.3, 23: 0.1
    }
    day_of_week_multipliers = {
        0: 1.0, 1: 1.0, 2: 1.1, 3: 1.0, 4: 1.2, 5: 1.5, 6: 1.3
    }
    day_of_month_multipliers = {
        1: 1.8, 15: 1.5, 28: 1.2, 29: 1.2, 30: 1.2, 31: 1.2
    }
    transaction_type_probabilities = {
        "Credit Card (POS/Online)": 0.60,
        "Online Purchase (Non-Card)": 0.15,
        "Venmo/PayPal Transfer": 0.15,
        "Bank Transfer (ACH)": 0.10
    }
    transaction_types = list(transaction_type_probabilities.keys())
    type_weights = list(transaction_type_probabilities.values())
    transaction_amount_ranges = {
        "Credit Card (POS/Online)": (5.00, 200.00),
        "Online Purchase (Non-Card)": (15.00, 500.00),
        "Venmo/PayPal Transfer": (10.00, 300.00),
        "Bank Transfer (ACH)": (50.00, 5000.00)
    }

    effective_active_hours = 12
    base_lambda = (avg_daily_transactions / effective_active_hours) / 3600

    transaction_id_counter = 0

    while current_time <= end_date:
        transaction_id_counter += 1

        hour_multiplier = hourly_frequency_multipliers.get(current_time.hour, 0.01)
        day_of_week_multiplier = day_of_week_multipliers.get(current_time.weekday(), 1.0)
        day_of_month_multiplier = day_of_month_multipliers.get(current_time.day, 1.0)

        combined_multiplier = hour_multiplier * day_of_week_multiplier * day_of_month_multiplier
        effective_lambda = base_lambda * combined_multiplier
        if effective_lambda <= 0:
            time_delta_seconds = 3600
        else:
            time_delta_seconds = np.random.exponential(scale=1/effective_lambda)

        if time_delta_seconds > timedelta(hours=3).total_seconds():
            time_delta_seconds = timedelta(hours=random.uniform(1, 3)).total_seconds()

        proposed_next_time = current_time + timedelta(seconds=time_delta_seconds)

        if proposed_next_time > end_date:
            break

        current_time = proposed_next_time

        # --- Select City with Home City Bias ---
        if random.random() < home_city_bias_probability:
            chosen_city = home_city
        else:
            if other_cities:
                chosen_city = random.choices(other_cities, weights=other_city_weights, k=1)[0]
            else:
                chosen_city = home_city

        chosen_type = random.choices(transaction_types, weights=type_weights, k=1)[0]

        entity_type_options = entity_pools_config[chosen_type]["entity_types"]
        chosen_entity_type = random.choice(entity_type_options)
        
        specific_entities_pool = entity_pools_config[chosen_type]["specific_entities"][chosen_entity_type]
        chosen_entity = random.choice(specific_entities_pool)

        min_amt, max_amt = transaction_amount_ranges.get(chosen_type, (1.00, 100.00))
        amount = round(random.uniform(min_amt, max_amt), 2)
        
        # --- New Accident Flags ---
        is_accident_burst = False
        is_high_amount_accident = False
        is_mistyped_entity_accident = False
        original_recipient_entity = chosen_entity # Default: original is the same as chosen


        # --- Simulate Accident Burst ---
        if random.random() < accident_probability:
            is_accident_burst = True # Mark the current transaction as triggering a burst


        # --- Simulate High Amount Accident (if not already part of a burst) ---
        if not is_accident_burst and random.random() < high_amount_accident_probability:
            is_high_amount_accident = True
            amount = round(max_amt * high_amount_multiplier * random.uniform(0.8, 1.2), 2)


        # --- Simulate Mistyped Entity Accident (if not already part of a burst) ---
        # Only apply to entities that are numeric or can be easily typo'd
        if not is_accident_burst and not is_high_amount_accident and random.random() < mistyped_entity_accident_probability:
            if chosen_entity_type in ["Person-to-Person", "Friend/Family Transfer"]:
                original_entity_str = str(chosen_entity)
                if original_entity_str.isdigit():
                    is_mistyped_entity_accident = True
                    # Store the original before modifying chosen_entity
                    original_recipient_entity = original_entity_str 
                    
                    num_digits_to_change = random.choice([1, 2])
                    modified_entity_list = list(original_entity_str)
                    
                    for _ in range(num_digits_to_change):
                        try:
                            idx = random.randint(0, len(modified_entity_list) - 1)
                            original_digit = int(modified_entity_list[idx])
                            new_digit = (original_digit + random.choice([-3, -2, -1, 1, 2, 3])) % 10
                            modified_entity_list[idx] = str(new_digit)
                        except ValueError:
                            is_mistyped_entity_accident = False
                            original_recipient_entity = chosen_entity # Revert if manipulation failed
                            break

                    if is_mistyped_entity_accident:
                        chosen_entity = "".join(modified_entity_list) # Update chosen_entity to be the mistyped one
                    else:
                        is_mistyped_entity_accident = False # Ensure flag is false if manipulation failed


        transactions.append({
            'customer_id': customer_id,
            'transaction_id': f"{customer_id}-{transaction_id_counter:08d}",
            'timestamp': current_time,
            'transaction_type': chosen_type,
            'amount': amount,
            'is_accident_burst': is_accident_burst,
            'is_high_amount_accident': is_high_amount_accident,
            'is_mistyped_entity_accident': is_mistyped_entity_accident,
            'city': chosen_city,
            'entity_type': chosen_entity_type,
            'recipient_entity': chosen_entity, # This is now the (potentially) mistyped one
            'original_recipient_entity': original_recipient_entity # New: The original, correct entity
        })

        # --- Simulate Accident Burst (sub-transactions) ---
        if is_accident_burst:
            num_accidents = random.randint(1, max_accident_transactions)
            for _ in range(num_accidents):
                accident_id_suffix = f"A{_ + 1}"
                accident_time_delta = random.uniform(0.01, 5.0)
                current_time += timedelta(seconds=accident_time_delta)

                if current_time > end_date:
                    break

                accident_type_options = ["Credit Card (POS/Online)", "Venmo/PayPal Transfer"]
                accident_chosen_type = random.choice(accident_type_options)

                accident_entity_type_options = entity_pools_config[accident_chosen_type]["entity_types"]
                accident_chosen_entity_type = random.choice(accident_entity_type_options)
                accident_specific_entities_pool = entity_pools_config[accident_chosen_type]["specific_entities"][accident_chosen_entity_type]
                accident_chosen_entity = random.choice(accident_specific_entities_pool)

                accident_is_high_amount = False
                accident_is_mistyped_entity = False
                accident_original_recipient_entity = accident_chosen_entity # For bursts, original is the same as chosen

                min_amt, max_amt = transaction_amount_ranges.get(accident_chosen_type, (1.00, 100.00))
                accident_amount = round(random.uniform(min_amt, max_amt), 2)

                transaction_id_counter += 1
                transactions.append({
                    'customer_id': customer_id,
                    'transaction_id': f"{customer_id}-{transaction_id_counter:08d}-{accident_id_suffix}",
                    'timestamp': current_time,
                    'transaction_type': accident_chosen_type,
                    'amount': accident_amount,
                    'is_accident_burst': True,
                    'is_high_amount_accident': accident_is_high_amount,
                    'is_mistyped_entity_accident': accident_is_mistyped_entity,
                    'city': chosen_city,
                    'entity_type': accident_chosen_entity_type,
                    'recipient_entity': accident_chosen_entity,
                    'original_recipient_entity': accident_original_recipient_entity
                })

    return pd.DataFrame(transactions)

# --- Example Usage for Multiple Customers with Home City and New Accident Types ---

start_date = datetime(2024, 1, 1, 0, 0, 0)
end_date = datetime(2024, 1, 20, 0, 0, 0)

num_customers = 100
all_customers_data = []
customer_home_cities = {}

print(f"Generating data for {num_customers} customers from {start_date} to {end_date}...")

for i in range(num_customers):
    current_customer_id = f"customer_{i+1:03d}"
    customer_home_city = random.choices(cities_all, weights=weights_all, k=1)[0]
    customer_home_cities[current_customer_id] = customer_home_city

    customer_avg_daily_transactions = random.randint(8, 15)
    customer_accident_probability = round(random.uniform(0.02, 0.05), 4)
    customer_home_city_bias = round(random.uniform(0.75, 0.95), 4)
    customer_high_amount_prob = round(random.uniform(0.003, 0.008), 4)
    customer_high_amount_multiplier = round(random.uniform(2.5, 4.0), 2)
    customer_mistyped_entity_prob = round(random.uniform(0.002, 0.005), 4)

    print(f"\n--- Generating for {current_customer_id} ---")
    print(f"  Home City: {customer_home_city} (Bias: {customer_home_city_bias:.2%})")
    print(f"  Average Daily Transactions: {customer_avg_daily_transactions}")
    print(f"  Burst Accident Probability: {customer_accident_probability:.2%}")
    print(f"  High Amount Accident Probability: {customer_high_amount_prob:.2%} (Multiplier: {customer_high_amount_multiplier}x)")
    print(f"  Mistyped Entity Accident Probability: {customer_mistyped_entity_prob:.2%}")


    customer_df = generate_synthetic_transactions(
        start_date=start_date,
        end_date=end_date,
        avg_daily_transactions=customer_avg_daily_transactions,
        accident_probability=customer_accident_probability,
        max_accident_transactions=random.randint(1, 3),
        high_amount_accident_probability=customer_high_amount_prob,
        high_amount_multiplier=customer_high_amount_multiplier,
        mistyped_entity_accident_probability=customer_mistyped_entity_prob,
        customer_id=current_customer_id,
        home_city=customer_home_city,
        home_city_bias_probability=customer_home_city_bias,
        cities_list_all=cities_all,
        city_selection_weights_all=weights_all,
        entity_pools_config=entity_pools
    )
    all_customers_data.append(customer_df)

synthetic_df = pd.concat(all_customers_data, ignore_index=True)

print(f"\n{'-'*50}\n")
print(f"Generated a total of {len(synthetic_df)} transactions across {num_customers} customers.")
print("\nFirst 10 transactions:")
print(synthetic_df.head(10))

print("\nLast 10 transactions:")
print(synthetic_df.tail(10))

print("\nDistribution of transactions by customer_id:")
print(synthetic_df['customer_id'].value_counts())

print("\nPercentage of transactions occurring in the customer's home city:")
home_city_df = pd.DataFrame.from_dict(customer_home_cities, orient='index', columns=['customer_home_city']).reset_index()
home_city_df.rename(columns={'index': 'customer_id'}, inplace=True)
df_with_home_city = pd.merge(synthetic_df, home_city_df, on='customer_id', how='left')
df_with_home_city['is_home_city_transaction'] = (df_with_home_city['city'] == df_with_home_city['customer_home_city'])
print(df_with_home_city.groupby('customer_id')['is_home_city_transaction'].mean().mul(100).round(2))
print(f"\nOverall percentage of transactions occurring in a customer's home city: {df_with_home_city['is_home_city_transaction'].mean() * 100:.2f}%")

# --- Analysis for Accident Types ---
print(f"\nTotal transactions marked as 'accident_burst': {synthetic_df['is_accident_burst'].sum()}")
print(f"Total transactions marked as 'high_amount_accident': {synthetic_df['is_high_amount_accident'].sum()}")
print(f"Total transactions marked as 'mistyped_entity_accident': {synthetic_df['is_mistyped_entity_accident'].sum()}")

print("\nSample High Amount Accidents:")
print(synthetic_df[synthetic_df['is_high_amount_accident']].head())

print("\nSample Mistyped Entity Accidents (showing original and mistyped):")
# Select relevant columns for this view
mistyped_sample = synthetic_df[synthetic_df['is_mistyped_entity_accident']][
    ['customer_id', 'timestamp', 'transaction_type', 'amount',
     'recipient_entity', 'original_recipient_entity', 'is_mistyped_entity_accident']
].head(10) # Show more than 5 to see a better sample
print(mistyped_sample)


print("\nDistribution of transactions by entity_type (top 5):")
print(synthetic_df['entity_type'].value_counts().head())

print("\nDistribution of transactions by recipient_entity (top 5):")
print(synthetic_df['recipient_entity'].value_counts().head())

#outputting the file
output_filename = "trial_transaction_data_3.csv"
print(f"Saving generated data to {output_filename}...")
synthetic_df.to_csv(output_filename, index=False)
print(f"Data saved successfully to {output_filename}")