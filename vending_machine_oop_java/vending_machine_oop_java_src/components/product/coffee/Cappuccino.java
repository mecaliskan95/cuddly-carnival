package components.product.coffee;

import components.product.Coffee;
import components.product.NutritionFacts;

public class Cappuccino extends Coffee { // Cappuccino is-a Coffee, therefore the class extends to Coffee.
	public Cappuccino() { // Constructor.
		super(3, "Cappuccino", 3.35, new NutritionFacts("1 coffee cup (6fl oz)", 56, 3.06, 4.36, 2.99));
	}

	// Declare methods.
	@Override
	public String texture() { // Overridden method.
		return "creamy";
	}

	@Override
	public String taste() { // Overridden method.
		return "strong sweet flavor";
	}
}