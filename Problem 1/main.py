# Initialize spot counts
vip_spots = 5
general_spots = 10

print("--- Welcome to the Parking Allocation System ---")

while True:
    name = input("\nEnter your name: ")
    is_vip = input("Are you a VIP? (yes/no): ").lower().strip() == "yes"

    if is_vip:
        if vip_spots > 0:
            vip_spots -= 1
            print(f"Welcome {name}! You have been assigned a VIP spot.")
        elif general_spots > 0:
            general_spots -= 1
            print(f"VIP spots full. {name}, you've been assigned a General spot.")
        else:
            print(f"Sorry {name}, the parking lot is completely full.")
    else:
        if general_spots > 0:
            general_spots -= 1
            print(f"Welcome {name}! You have been assigned a General spot.")
        else:
            print(f"Sorry {name}, general parking is full.")

    # Status Update
    print(f"Status: {vip_spots} VIP left | {general_spots} General left")

# Check if everything is full to close the system
    if vip_spots == 0 and general_spots == 0:
        print("\n--- The parking lot is now completely full. System shutting down. ---")
        break