package components;

import components.product.bakery.Bagel;
import components.product.bakery.Croissant;
import components.product.coffee.Cappuccino;
import components.product.coffee.Espresso;
import components.product.coffee.Latte;

public class Inventory {
	// Declare variables.
	private int qtyLatte; // Latte quantity in pcs
	private int qtyEspresso; // Espresso quantity in pcs
	private int qtyCappucino; // Cappucino quantity in pcs
	private int qtyBagel; // Bagel quantity in pcs
	private int qtyCroissant; // Croissant quantity in pcs

	// Declare methods.
	public int getQtyLatte() { // Getter for Latte quantity.
		return qtyLatte;
	}

	public int getQtyEspresso() { // Getter for Espresso quantity.
		return qtyEspresso;
	}

	public int getQtyCappucino() { // Getter for Cappucino quantity.
		return qtyCappucino;
	}

	public int getQtyBagel() { // Getter for Bagel quantity.
		return qtyBagel;
	}

	public int getQtyCroissant() { // Getter for Croissant quantity.
		return qtyCroissant;
	}

	public void setQtyLatte(int qtyLatte) { // Setter for Latte quantity.
		this.qtyLatte = qtyLatte;
	}

	public void setQtyEspresso(int qtyEspresso) { // Setter for Espresso quantity.
		this.qtyEspresso = qtyEspresso;
	}

	public void setQtyCappucino(int qtyCappucino) { // Setter for Cappucino quantity.
		this.qtyCappucino = qtyCappucino;
	}

	public void setQtyBagel(int qtyBagel) { // Setter for Bagel quantity.
		this.qtyBagel = qtyBagel;
	}

	public void setQtyCroissant(int qtyCroissant) { // Setter for Croissant quantity.
		this.qtyCroissant = qtyCroissant;
	}

	public boolean inStock(Product product) { // Checks and returns if a product is instock.
		if (product instanceof Latte) {
			if (qtyLatte > 0) {
				return true;
			} else {
				return false;
			}
		} else if (product instanceof Espresso) {
			if (qtyEspresso > 0) {
				return true;
			} else {
				return false;
			}
		} else if (product instanceof Cappuccino) {
			if (qtyCappucino > 0) {
				return true;
			} else {
				return false;
			}
		} else if (product instanceof Bagel) {
			if (qtyBagel > 0) {
				return true;
			} else {
				return false;
			}
		} else if (product instanceof Croissant) {
			if (qtyCroissant > 0) {
				return true;
			} else {
				return false;
			}
		} else
			return false;
	}

	public void removeProduct(Register register, Product product) {
		if (!register.isCancel()) {
			if (product == null) {
			} else {
				// Remove a quantity of the product from the inventory.
				if (product instanceof Latte) {
					setQtyLatte(getQtyLatte() - 1);
				} else if (product instanceof Espresso) {
					setQtyEspresso(getQtyEspresso() - 1);
				} else if (product instanceof Cappuccino) {
					setQtyCappucino(getQtyCappucino() - 1);
				} else if (product instanceof Bagel) {
					setQtyBagel(getQtyBagel() - 1);
				} else if (product instanceof Croissant) {
					setQtyCroissant(getQtyCroissant() - 1);
				}
				System.out.println();
			}
		}
	}

	public void addProduct(Panel panel) {
		int n;
		panel.displayRestockMenu();
		while (true) { // Restock sequence for Latte.
			panel.promptAddLatte();
			n = panel.readIntEntry();
			if (n >= 0) {
				setQtyLatte(getQtyLatte() + n);
				break;
			} else {
				panel.displayInvalidAction();
			}
		}
		while (true) { // Restock sequence for Espresso.
			panel.promptAddEspresso();
			n = panel.readIntEntry();
			if (n >= 0) {
				setQtyEspresso(getQtyEspresso() + n);
				break;
			} else {
				panel.displayInvalidAction();
			}
		}
		while (true) { // Restock sequence for Cappucino.
			panel.promptAddCappucino();
			n = panel.readIntEntry();
			if (n >= 0) {
				setQtyCappucino(getQtyCappucino() + n);
				break;
			} else {
				panel.displayInvalidAction();
			}
		}
		while (true) { // Restock sequence for Bagel.
			panel.promptAddBagel();
			n = panel.readIntEntry();
			if (n >= 0) {
				setQtyBagel(getQtyBagel() + n);
				break;
			} else {
				panel.displayInvalidAction();
			}
		}
		while (true) { // Restock sequence for Croissant.
			panel.promptAddCroissant();
			n = panel.readIntEntry();
			if (n >= 0) {
				setQtyCroissant(getQtyCroissant() + n);
				break;
			} else {
				panel.displayInvalidAction();
			}
		}
	}
}