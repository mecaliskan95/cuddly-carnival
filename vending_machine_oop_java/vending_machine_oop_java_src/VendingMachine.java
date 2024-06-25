// Declare package.


// Import libraries.
import java.util.ArrayList;
import java.util.List;

import components.Inventory;
import components.Memory;
import components.Panel;
import components.Product;
import components.Register;
import components.product.bakery.Bagel;
import components.product.bakery.Croissant;
import components.product.coffee.Cappuccino;
import components.product.coffee.Espresso;
import components.product.coffee.Latte;

public class VendingMachine {

	// Main method.
	public static void main(String[] args) {

		// Create an object of each type.
		Panel panel = new Panel();
		Register register = new Register();
		Inventory inventory = new Inventory();
		Memory memory = new Memory();
		Product product = null;

		bootUp(panel);
		// Vending machine boots up.
		while (true) {
			panel.displayMainMenu();
			panel.promptSelection();
			int m = panel.readIntEntry();
			switch (m) {
			case 1: // Customer sequence.
				sellProduct(panel, register, inventory, product); // Sells a product.
				break;
			case 2: // Admin sequence.
				panel.promptPassword(); // Asks for the Password.
				boolean isAdmin = memory.verifyPassword(panel); // Verifies the Password for Admin.
				while (isAdmin) { // If the password is correct, then the Admin menu is accessed.
					boolean isBack = false; // Triggers to return back to main menu.
					do {
						panel.displayAdminMenu();
						switch (panel.readIntEntry()) {
						case 1: // Changes password
							changePassword(memory, panel);
							break;
						case 2: // Views the remaining inventory
							viewRemainingInventory(inventory, panel);
							break;
						case 3: // Allows to restock inventory
							restockInventory(inventory, panel);
							break;
						case 4: // Views the stored cash in the register
							viewStoredCash(register, panel);
							break;
						case 5: // Allows to withdraw cash
							withdrawCash(register, panel);
							break;
						case -1: // Returns back to the main menu.
							isBack = true;
							isAdmin = false;
							break;
						default: // Out-of-range case selected
							panel.displayInvalidAction();
							break;
						}
					} while (!isBack);
				}
				break;
			case -1: // to Exit
				exitSystem(panel);
				// Complete Shut-down. All changed properties of objects such as Inventory,
				// Register & Memory will be reset.
			default:
				panel.displayInvalidAction();
				break;
			}
		}
	}

	// Declare methods.
	public static void changePassword(Memory memory, Panel panel) { // Changes the password.
		panel.promptNewPassword();
		memory.setPassword(panel.readIntEntry(), panel);
	}

	public static void viewRemainingInventory(Inventory inventory, Panel panel) { // Views the remaining inventory.
		panel.displayRemainingInventory(inventory);
	}

	public static void restockInventory(Inventory inventory, Panel panel) { // Restocks the inventory.
		inventory.addProduct(panel);
		panel.displayReturningtoMenu();
	}

	public static void withdrawCash(Register register, Panel panel) { // Withdraws Cash.
		panel.displayStoredCash(register.getStoredCash());
		double n = 0;
		while (true) {
			panel.promptWithdrawCash();
			n = panel.readDoubleEntry();
			if (n == -1) { // Cancels the action.
				panel.displayActionCancelled();
				break;
			} else if ((register.getStoredCash() - n < 0) || (n < 0)) {
				panel.displayInvalidAction();
			} else {
				register.setStoredCash(register.getStoredCash() - n);
				panel.displayStoredCash(register.getStoredCash());
				break;
			}
		}
	}

	public static void viewStoredCash(Register register, Panel panel) { // Views stored cash.
		panel.displayStoredCash(register.getStoredCash());
		panel.displayReturningtoMenu();
	}

	public static void bootUp(Panel panel) {
		// Starts-up the vending machine.
		int n;
		panel.displayEnterKeytoStart();
		do {
			n = panel.readIntEntry();
			if (n == -1) {
				exitSystem(panel);
			} else if (n != 1) {
				panel.displayInvalidAction();
				panel.promptSelection();
			}
		} while (n != 1);
	}

	public static void exitSystem(Panel panel) { // Exit.
		panel.displayExitSystem();
		System.exit(0);
	}

	public static void sellProduct(Panel panel, Register register, Inventory inventory, Product product) {
		// Product Selection Sequence.

		// Create a list of products.
		List<Product> productList = new ArrayList<>();
		productList.add(new Latte());
		productList.add(new Espresso());
		productList.add(new Cappuccino());
		productList.add(new Bagel());
		productList.add(new Croissant());

		boolean isProceed = false;
		do {

			panel.displayProducts(productList, inventory);
			boolean isBack = false;
			do {

				panel.promptSelection();
				int n = panel.readIntEntry();
				switch (n) {
				case 1: // Product -> Latte polymorphism applied.
					product = new Latte();
					panel.displaySelection(product, inventory);
					break;
				case 2: // Product -> Espresso polymorphism applied.
					product = new Espresso();
					panel.displaySelection(product, inventory);
					break;
				case 3: // Product -> Cappuccino polymorphism applied.
					product = new Cappuccino();
					panel.displaySelection(product, inventory);
					break;
				case 4: // Product -> Bagel polymorphism applied.
					product = new Bagel();
					panel.displaySelection(product, inventory);
					break;
				case 5: // Product -> Croissant polymorphism applied.
					product = new Croissant();
					panel.displaySelection(product, inventory);
					break;
				case -1: // Return back to main menu.
					product = null;
					isBack = true;
					break;
				default: // Returns null to allow re-selection.
					product = null;
					panel.displayInvalidAction();
					break;
				}
			} while (!inventory.inStock(product) && isBack == false);
			if (product == null) {
				break;
			} else {
				panel.displayInfo(product, product.getName());
				panel.promptProceed(product.getName());
				while (true) {
					int n = panel.readIntEntry();
					if (n == 1) {
						isProceed = true;
						break;
					} else if (n == -1) {
						isProceed = false;
						break;
					} else {
						panel.displayInvalidAction();
						panel.promptSelection();
					}
				}
			}
		} while (!isProceed);
		if (product == null) {
			// This is to return back to menu when [-1] is triggered and no product is
			// selected.
			panel.displayReturningtoMenu();
		} else {
			panel.displaySummary(product.getPrice(), product.getName());
			// Payment Sequence.
			register.processPayment(panel, product.getPrice());
			register.returnCash(product.getPrice(), panel);
			// Dispense the Product.
			if(!register.isCancel()) {
				inventory.removeProduct(register, product);
				panel.displayDispensedProduct(product.getName());
			}
		}
	}
}