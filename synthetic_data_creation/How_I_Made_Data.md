# Synthetic Transaction Data Generator: How I Cooked Up This Data!

Hey there! So, you've got this super cool synthetic transaction data, right? And you're probably wondering, "How did this magic happen? What's under the hood?" Well, let me spill the beans on how I went about creating this bad boy.

The whole point here was to churn out some realistic-ish transaction data. Not just random numbers, but stuff that actually looks and feels like real-world financial activity. This kind of data is prime for training machine learning models, especially for sniffing out weird stuff like **anomalies** or potential **fraud**.

## The Master Plan: Customer-First, Time-Aware, Anomaly-Injected

My core idea was to build this data from the ground up, customer by customer, making sure it behaved like actual people doing actual transactions over time. And, because life (and fraud) isn't always smooth sailing, I deliberately injected some "accidents" and unusual events.

Here’s the breakdown of how I did it:

### 1. Generating Transactions, Customer by Customer

I set up a loop to go through a bunch of customers – for our current big run, we're doing around **900 customers** to hit that sweet spot of about **1 million transactions** over a three-month period (January 1st to March 31st, 2024).

For *each individual customer*, I randomized a bunch of parameters to make them unique, just like real people are:

* **Average Daily Transactions:** Every customer got their own vibe, ranging from `8 to 15 transactions per day`. Some folks just spend more, ya know?
* **Home Sweet Home City Bias:** Each customer got assigned a `home city` (picked based on real US city population weights – fancy!). Most of their transactions (`75% to 95%`) lean towards this home city, but they'll still pop up in other cities sometimes. Because, hey, people travel!
* **Per-Customer "Accident" Probabilities:** This is a big one! Instead of a global chance for weird stuff, **every single customer got their own unique probability** for each type of "accident" we injected. So, `customer_001` might be super prone to one type of hiccup, while `customer_002` hardly ever sees it. This makes the data much more realistic and challenging for models.

### 2. Making Transactions Feel Real (Time & Type-Wise)

Transactions don't just happen randomly. They've got a rhythm. I baked that into the generation process:

* **Hourly Patterns:** More transactions during business hours (9 AM - 5 PM), lunchtime rush (12-2 PM), and after-work/dinner time (5-8 PM). Not much happening at 3 AM unless it's an online binge!
* **Day of Week:** Weekends and Fridays often see a bump in activity.
* **Day of Month:** Beginning and end of the month (payday, bill due dates) often have more action.
* **Transaction Types & Amounts:** I made sure transactions had different `types` (Credit Card, Online Purchase, Venmo/PayPal, Bank Transfer) with realistic probabilities and amount ranges. You wouldn't pay rent with a credit card at the grocery store, right? And a Venmo payment usually isn't for $5000.
* **Recipient Entities:** This is where it gets detailed! Each transaction type has its own pool of `entity types` (like "Grocery," "Restaurant," "Person-to-Person") and then `specific entities` within those types (like "SuperMart," "Amazon," or even "user_1234567890"). This makes the `recipient_entity` field look super genuine.

### 3. Injecting the "Accidents" (The Fun Part for Anomaly Detection!)

This is where the data becomes truly valuable for training. I specifically added three types of "accidents" to simulate real-world oddities:

* **`is_accident_burst` (Back-to-Back Transactions):**
    * **What it is:** Imagine hitting "confirm" twice on your phone, or a faulty card reader splitting one purchase into three. That's a burst! These are multiple transactions (1 to 3 extra) happening within seconds of each other.
    * **Why it's important:** It could be a simple user error, a system glitch, or even a fraudster quickly testing a stolen card. Your models need to spot these rapid-fire events.
    * **How it works:** Each customer has a small `accident_probability` for a normal transaction to suddenly trigger a burst.

* **`is_high_amount_accident` (Unusually High Amount):**
    * **What it is:** A transaction that's significantly larger than the usual maximum for that transaction type. Like buying a private jet on your grocery card.
    * **Why it's important:** This is a classic fraud indicator. But it could also be a legitimate, rare big purchase (like buying a car). Your model needs to differentiate.
    * **How it works:** Each customer has a `high_amount_accident_probability` for this to happen. If it does, the amount is multiplied by a `high_amount_multiplier` (2.5x to 4x the max normal amount) to make it stand out.

* **`is_mistyped_entity_accident` (Mistyped Recipient Entity):**
    * **What it is:** This is when you try to send money to "user 1234567890" but accidentally type "user 1234567891". Just a digit or two off.
    * **Why it's important:** Super common user error, especially with manual entry for peer-to-peer (P2P) or bank transfers. Sometimes, though, it could be part of a scam if a user is tricked into sending money to a slightly wrong account.
    * **The Big Detail for Training:** When this happens, the `recipient_entity` column actually contains the *mistyped* value. But here's the cool part for you: I added an `original_recipient_entity` column that stores the **correct, intended value**! This is golden for training because your model can learn to identify that "wrong" entity and, if needed, even figure out what the "right" one was supposed to be.

### 4. Outputting the Gold: Straight to AWS S3!

Instead of cluttering up my local drive with a massive CSV file, I wired this whole setup to dump the data directly into your AWS S3 bucket.

* **No Local Files:** Once the DataFrame is built in memory, it doesn't touch the local disk. This is way faster and cleaner for big datasets like our million-row monster.
* **In-Memory Magic:** I use Python's `io.StringIO` to basically create a fake "file" in memory, write the DataFrame to that as a CSV, and then `boto3` grabs that in-memory content and shoves it right up to S3 using the `put_object` method.
* **Your Destination:** The data should now be chilling at `s3://accidentaltransferproject/data/synthetic_transactions_labeled.csv`.

So, yeah, that's the long and short of it! This data isn't just random, it's got layers of realism and carefully injected anomalies, all designed to give your models a serious workout. Go forth and train!