package components;

public class Register {

	// Declare variables.
	private double storedCash; // Variable for the stored cash inside the vending machine.
	private double insertedCash; // Variable for the inserted cash inside the vending machine. Must reset to 0
									// after each transaction.
	private boolean isCancel; // Variable for the statement of payment process/ cancellation.

	// Declare methods.
	public void processPayment(Panel panel, double price) { // Processes payments/ transactions.
		do {
			panel.displayTotalCash(getInsertedCash());
			double cash = panel.readDoubleEntry();
			if (cash == -1) {
				isCancel = true;
			} else if (cash < 0) {
				panel.displayInvalidAction();
				isCancel = false;
			} else {
				insertedCash += cash;
				isCancel = false;
			}
		} while (insertedCash < price && isCancel == false);
		if (!isCancel) {
			panel.displayPaymentProcessed();
		} else {
			panel.displayPaymentCancelled();
		}
	}

	public void returnCash(double price, Panel panel) { // Calculates the cash return.
		if (!isCancel) {
			storedCash += price;
			insertedCash -= price;
		}
		panel.displayReturnCash(insertedCash);
		insertedCash = 0;
	}

	public double getStoredCash() { // Getter for stored cash.
		return storedCash;
	}

	public double getInsertedCash() { // Getter for inserted cash.
		return insertedCash;
	}

	public void setStoredCash(double storedCash) { // Setter for stored cash.
		this.storedCash = storedCash;
	}

	public boolean isCancel() { // Getter for isCancel.
		return isCancel;
	}

}