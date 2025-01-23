import datetime

def display_menu():
    print("\nWelcome to FinanceBot! How can I assist you today?")
    print("1. Where to invest for best returns")
    print("2. How long to invest")
    print("3. Tips for a diversified portfolio")
    print("4. Exit")

def get_investment_advice():
    print("\nLet me help you find the best investment options!")
    risk_profile = input("What is your risk tolerance? (low/medium/high): ").strip().lower()
    amount = float(input("How much do you plan to invest? ($): "))

    if risk_profile == "low":
        print("\nRecommended investments:")
        print("- Government bonds")
        print("- High-yield savings accounts")
        print("- Index funds")
    elif risk_profile == "medium":
        print("\nRecommended investments:")
        print("- Balanced mutual funds")
        print("- ETFs (Exchange Traded Funds)")
        print("- Real estate investment trusts (REITs)")
    elif risk_profile == "high":
        print("\nRecommended investments:")
        print("- Individual stocks")
        print("- Cryptocurrency")
        print("- Startups or venture capital")
    else:
        print("Invalid risk profile. Please choose low, medium, or high.")

    print(f"\nBased on your investment of ${amount:,.2f}, consider diversifying across these options.")

def get_investment_duration():
    print("\nDetermining the ideal investment duration.")
    goal = input("What is your financial goal? (e.g., buying a house, retirement, vacation): ").strip().lower()

    if goal in ["buying a house", "house"]:
        print("Recommended duration: 5-10 years (medium-term investments).")
    elif goal == "retirement":
        print("Recommended duration: 10+ years (long-term investments).")
    elif goal == "vacation":
        print("Recommended duration: 1-3 years (short-term investments).")
    else:
        print("For general goals, consider your timeline and risk tolerance.")

def get_diversification_tips():
    print("\nHere are some tips for a diversified portfolio:")
    print("1. Allocate assets across stocks, bonds, and cash.")
    print("2. Invest in different industries and sectors.")
    print("3. Consider international investments for geographic diversity.")
    print("4. Rebalance your portfolio periodically.")

def main():
    while True:
        display_menu()
        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == "1":
            get_investment_advice()
        elif choice == "2":
            get_investment_duration()
        elif choice == "3":
            get_diversification_tips()
        elif choice == "4":
            print("\nThank you for using FinanceBot. Have a great day!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
