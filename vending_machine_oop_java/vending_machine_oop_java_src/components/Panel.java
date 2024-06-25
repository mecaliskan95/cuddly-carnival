package components;

// Import libraries.
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.InputMismatchException;
import java.util.Scanner;

public class Panel { // This class is the panel of the Vending Machine. The functionality is limited
						// only to the print functions and reading the user input.

	// Declare methods
	public void displayMainMenu() {
		try { // Reads a txt .file. Used an ASCII Art to be creative.
			FileReader reader = new FileReader(
					"D:\\Lectures\\Hacettepe University - BBS515 - Object-Oriented Programming\\Project\\BBS515Proje_N22233552_Mert_Caliskan\\src\\components\\Welcome.txt");
			int data = reader.read();
			while (data != -1) {
				System.out.print((char) data);
				data = reader.read();
			}
			reader.close();

		} catch (FileNotFoundException e) { // Catch if FileNotFoundException is occured.
			System.out.println(
					"FileNotFoundException occured while loading the 'Welcome' banner. Please check. Proceeding...");
			// Notify the user but still proceed with the exception.
		} catch (IOException e) {
			System.out.println("IOException occured while loading the 'Welcome' banner. Please check. Proceeding...");
			// Notify the user but still proceed with the exception.
		}
		System.out.println("\n[            WELCOME TO THE VENDING MACHINE            ]");

		// Added local date and time to the welcome menu.
		LocalDateTime datetime1 = LocalDateTime.now();
		DateTimeFormatter dateFormat = DateTimeFormatter.ofPattern("dd/MM/yyyy");
		DateTimeFormatter timeFormat = DateTimeFormatter.ofPattern("HH:mm");
		String formatDate = datetime1.format(dateFormat);
		String formatTime = datetime1.format(timeFormat);
		System.out.println("Date: " + formatDate + "\t\t\t     Time: " + formatTime);

		// Main Menu.
		System.out.println("Please select your credentials.");
		System.out.println("-> [1]\tCustomer");
		System.out.println("-> [2]\tAdministrator");
		System.out.println("-> [-1]\tSystem Exit");
	}

	public void displayProducts(List<Product> productList, Inventory inventory) { 
		// Displays the products. Preferred to create a List here.
		System.out.println("   \033[4mID\033[0m\t\033[4mName\033[0m\t\t\t\033[4mPrice\033[0m\t\033[4mIn Stock\033[0m");
		for (Product product : productList) {
			System.out.println("-> [" + product.getId() + "]" + "\t" + product.getName() + "\t\t$" + product.getPrice()
					+ "\t" + inventory.inStock(product));
		}
		System.out.println("-> [-1]\tBack");
	}

	public void displaySelection(Product product, Inventory inventory) { // Displays the selected product and its
																			// defined properties.
		if (inventory.inStock(product)) {
			System.out.println("You have selected [" + product.getId() + "] for " + product.getName() + ".");
			System.out.println(
					product.getName() + " has " + product.texture() + " texture, smells like " + product.smell()
							+ " and has " + product.taste() + " taste. It feels " + product.temperature() + ".");
		} else {
			System.out.println("Your selection: [" + product.getId() + "] - " + product.getName()
					+ " is currently out of stock. Please select another product or request an Administrator to restock.");
		}
	}

	public void promptSelection() { // Ask for a selection.
		System.out.print("Please enter your selection: ");
	}

	public void displayInfo(Product product, String name) { // Prompts and displays nutrition
															// facts of the selected product.
		System.out.println("Would you like to display the nutrition facts for " + name + "?");
		System.out.println("-> [1]\tYes");
		System.out.println("-> [-1]\tNo");
		promptSelection();
		while (true) {
			int n = readIntEntry();
			if (n == 1) {
				System.out.println(
						"Nutrition facts for [" + product.getId() + "] " + product.getName() + " are given below.");
				System.out.println("\033[4mServing Size:\t" + product.getNutritionFacts().getServingSize()+"\033[0m");
				System.out.println("Calories:\t" + product.getNutritionFacts().getCalories() + " cal.");
				System.out.println("Protein:\t" + product.getNutritionFacts().getProtein() + " gr.");
				System.out.println("Fat:\t\t" + product.getNutritionFacts().getFat() + " gr.");
				System.out.println("Carbs:\t\t" + product.getNutritionFacts().getCarbs() + " gr.");
				break;
			} else if (n == -1)
				break;
			else {
				displayInvalidAction();
				promptSelection();
			}
		}
	}

	public void promptProceed(String name) { // Asks the user if they would like to continue with their selection.
		System.out.println("Would you like to proceed with your selection: " + name + "?");
		System.out.println("-> [1]\tProceed");
		System.out.println("-> [-1]\tRe-select");
		promptSelection();
	}

	public void displaySummary(double n, String name) { // Displays the total price for the selection before payment
														// sequence.
		System.out.println("Your total is $" + n + " for " + name + ".");
	}

	public void displayTotalCash(double n) { // Displays the total cash inserted.
		System.out.print("You have inserted a total of $" + n
				+ ". Please insert more cash for payment. Enter [-1] to cancel. $");
	}

	public void displayReturnCash(double n) { // Displays the cash return amount.
		System.out.printf("Your return is $%.2f. Please pick it up from the register.\n", n);
	}

	public void displayPaymentProcessed() { // Displays if the payment is processed.
		System.out.println("Your payment is processed...");
	}

	public void displayPaymentCancelled() { // Display if the payment is cancelled by entering [-1].
		System.out.println("Your payment is cancelled...");
	}

	public void displayDispensedProduct(String name) { 
		// Displays if the payment is successful and the product is dispensed.
		System.out.println("Your " + name + " is dispensed. Please pick it up from the dispenser. Enjoy!");
	}

	public void displayInvalidAction() { // Displays if the user made an invalid entry.
		System.out.println("You have made an invalid action. Please try again.");
	}

	public void displayAccessGranted() { // Displays access granted message for Administrator access.
		System.out.println("Access granted!");
	}

	public void displayIncorrectPassword() { // Displays if the entered password is incorrect.
		System.out.println("Incorrect password. Please try again.");
		promptPassword();
	}

	public void promptPassword() { // Asks for the Administrator password or cancel.
		System.out.print("Please enter your password or enter [-1] to cancel: ");
	}

	public void displayAdminMenu() { // Displays the Administrator menu.
		System.out.println("Please make your selection.");
		System.out.println("-> [1]\tChange Password");
		System.out.println("-> [2]\tView Remaining Inventory");
		System.out.println("-> [3]\tRestock Inventory");
		System.out.println("-> [4]\tView Stored Cash");
		System.out.println("-> [5]\tWithdraw Cash");
		System.out.println("-> [-1]\tBack to Main Menu");
		promptSelection();
	}

	public void displayRemainingInventory(Inventory inventory) { // Displays the remaining inventory.
		System.out.println("The current inventory is listed below.");
		System.out.println("-> Caffe Latte:\t\t" + inventory.getQtyLatte() + " pcs");
		System.out.println("-> Espresso:\t\t" + inventory.getQtyEspresso() + " pcs");
		System.out.println("-> Cappuccino:\t\t" + inventory.getQtyCappucino() + " pcs");
		System.out.println("-> Plain Bagel:\t\t" + inventory.getQtyBagel() + " pcs");
		System.out.println("-> Choc. Croissant:\t" + inventory.getQtyCroissant() + " pcs");
		displayReturningtoMenu();
	}

	public void displayActionCancelled() { // Displays if an action is cancelled by entering [-1].
		System.out.println("Action is cancelled. ");
		displayReturningtoMenu();
	}

	public void displayRestockMenu() { // Displays restock inventory message.
		System.out.println("Please enter the amount of inventory you would like to add in pcs.");
	}

	public void promptAddLatte() { // Asks for the amount of Caffe Latte to be added.
		System.out.print("-> Caffe Latte:\t\tpcs ");
	}

	public void promptAddEspresso() { // Asks for the amount of Espresso to be added.
		System.out.print("-> Espresso:\t\tpcs ");
	}

	public void promptAddCappucino() { // Asks for the amount of Cappucino to be added.
		System.out.print("-> Cappucino:\t\tpcs ");
	}

	public void promptAddBagel() { // Asks for the amount of Plain Bagel to be added.
		System.out.print("-> Plain Bagel:\t\tpcs ");
	}

	public void promptAddCroissant() { // Asks for the amount of Chocolate Croissant to be added.
		System.out.print("-> Choc. Croissant:\tpcs ");
	}

	public void promptWithdrawCash() { // Asks for the amount of Cash to be withdrawn.
		System.out.print("Please enter the amount of cash you would like to withdraw or enter [-1] to cancel. $");
	}

	public void displayStoredCash(double n) { // Displays the stored cash in the inventory.
		System.out.printf("The vending machine currently has a total of $%.2f.\n", n);
	}

	public void promptCash() { // Asks the User to enter a valid cash amount.
		System.out.print("Please enter a valid cash amount. $");
	}

	public void promptNewPassword() { // Asks for the new password or cancel.
		System.out.print(
				"Your password must be made of numbers and can NOT be longer than 8 digits. Please enter the new password or [-1] to cancel: ");
	}

	public void displayPasswordChanged(int n) { // Displays the new password.
		System.out.println("You have successfully changed the password to <<" + n + ">>");
	}

	public void displayReturningtoMenu() { // Displays when returning back to the menu.
		System.out.println("Returning back to the menu...");
	}

	public void displayEnterKeytoStart() { // Displays on boot-up.
		System.out.print("Please enter [1] to start up the vending machine or [-1] to exit system... ");
	}

	public void displayExitSystem() { // Displays when exit is triggered.
		System.out.println("You have chosen to exit the system. Thank you for using the vending machine!");
	}

	public int readIntEntry() {
		// Reads int input.
		int n = 0;
		Scanner scanner = new Scanner(System.in);
		boolean isProceed = false;
		do {
			try {
				n = scanner.nextInt();
				isProceed = true;
			} catch (InputMismatchException e) {
				scanner.next();
				displayInvalidAction();
				promptSelection();
				isProceed = false;
			}
		} while (!isProceed);
		return n;
	}

	public double readDoubleEntry() {
		// Reads double input.
		double n = 0;
		Scanner scanner = new Scanner(System.in);
		boolean isProceed = false;
		do {
			try {
				n = scanner.nextDouble();
				isProceed = true;
			} catch (InputMismatchException e) {
				scanner.next();
				displayInvalidAction();
				promptCash();
				isProceed = false;
			}
		} while (!isProceed);
		return n;
	}
}