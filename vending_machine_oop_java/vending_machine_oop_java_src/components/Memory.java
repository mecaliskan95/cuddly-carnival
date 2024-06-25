package components;

public class Memory {
	// Declare variables
	private int password = 1234; // Password by default.

	// Declare methods
	public int getPassword() { // Getter for password.
		return password;
	}

	public void setPassword(int password, Panel panel) { // Setter for password.
		while (true) {
			if (password == -1) {
				panel.displayActionCancelled();
				break;
			} else if (password < -1) {
				panel.displayInvalidAction();
			} else {
				this.password = password;
				panel.displayPasswordChanged(getPassword());
				break;
			}
		}
	}

	public boolean verifyPassword(Panel panel) { // Verifies the entered password.
		while (true) {
			int n = panel.readIntEntry();
			if (getPassword() == n) {
				panel.displayAccessGranted();
				return true;
			} else if (n == -1) {
				panel.displayActionCancelled();
				return false;
			} else {
				panel.displayIncorrectPassword();
			}
		}
	}
}