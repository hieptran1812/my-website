const nodemailer = require("nodemailer");
require("dotenv").config({ path: ".env.local" });

async function testEmail() {
  try {
    console.log("Testing email configuration...");
    console.log("Gmail User:", process.env.GMAIL_USER);
    console.log(
      "Gmail App Password configured:",
      !!process.env.GMAIL_APP_PASSWORD
    );

    const transporter = nodemailer.createTransport({
      service: "gmail",
      auth: {
        user: process.env.GMAIL_USER,
        pass: process.env.GMAIL_APP_PASSWORD,
      },
    });

    // Verify the transporter
    console.log("Verifying transporter...");
    await transporter.verify();
    console.log("✅ Transporter verified successfully!");

    // Send test email
    console.log("Sending test email...");
    const result = await transporter.sendMail({
      from: process.env.GMAIL_USER,
      to: "hieptran.jobs@gmail.com",
      subject: "Test Contact Form - Email Setup Verification",
      html: `
        <h2>Contact Form Test</h2>
        <p>This is a test email to verify that the contact form email functionality is working correctly.</p>
        <p><strong>Timestamp:</strong> ${new Date().toISOString()}</p>
        <p><strong>From:</strong> Contact form test script</p>
      `,
    });

    console.log("✅ Email sent successfully!");
    console.log("Message ID:", result.messageId);
  } catch (error) {
    console.error("❌ Email test failed:", error.message);
    console.error("Full error:", error);
  }
}

testEmail();
