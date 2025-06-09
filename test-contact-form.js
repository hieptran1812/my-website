// Test script to check contact form API
const testContactForm = async () => {
  const testData = {
    name: "Test User",
    email: "test@example.com",
    subject: "Test Subject",
    message: "This is a test message to verify the contact form is working.",
  };

  try {
    console.log("Testing contact form API...");
    console.log("Sending test data:", testData);

    const response = await fetch("http://localhost:3001/api/contact", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(testData),
    });

    console.log("Response status:", response.status);

    const result = await response.json();
    console.log("Response body:", result);

    if (response.ok) {
      console.log("✅ Contact form API is working!");
    } else {
      console.log("❌ Contact form API failed:", result.error);
    }
  } catch (error) {
    console.error("❌ Error testing contact form:", error.message);
  }
};

testContactForm();
