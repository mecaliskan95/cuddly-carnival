package components.product;

public class NutritionFacts {
	// Declare variables.
	private String servingSize; // to be defined based on the product type.
	private double calories; // in cal.
	private double protein; // in grams.
	private double fat; // in grams.
	private double carbs; // in grams.

	// All-args constructor.
	public NutritionFacts(String servingSize, double calories, double protein, double fat, double carbs) {
		this.servingSize = servingSize;
		this.calories = calories;
		this.protein = protein;
		this.fat = fat;
		this.carbs = carbs;
	}

	// Declare methods.
	public String getServingSize() { // Getter for serving size
		return servingSize;
	}

	public double getCalories() { // Getter for calories
		return calories;
	}

	public double getProtein() { // Getter for protein
		return protein;
	}

	public double getFat() { // Getter for fat
		return fat;
	}

	public double getCarbs() { // Getter for carbs
		return carbs;
	}
}