import { NextRequest, NextResponse } from "next/server";
import nodemailer from "nodemailer";

export async function POST(request: NextRequest) {
  try {
    const { name, email, subject, message } = await request.json();

    // Validation
    if (!name || !email || !subject || !message) {
      return NextResponse.json(
        { error: "All fields are required" },
        { status: 400 }
      );
    }

    // Email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      return NextResponse.json(
        { error: "Please provide a valid email address" },
        { status: 400 }
      );
    }

    // Check environment variables
    if (!process.env.GMAIL_USER || !process.env.GMAIL_APP_PASSWORD) {
      console.error("Missing Gmail configuration in environment variables");
      return NextResponse.json(
        { error: "Email service is not configured properly. Please contact the administrator." },
        { status: 500 }
      );
    }

    // Create nodemailer transporter using Gmail
    const transporter = nodemailer.createTransport({
      service: "gmail",
      auth: {
        user: process.env.GMAIL_USER,
        pass: process.env.GMAIL_APP_PASSWORD,
      },
    });

    // Email content
    const mailOptions = {
      from: process.env.GMAIL_USER,
      to: "hieptran.jobs@gmail.com",
      subject: `Contact Form: ${subject}`,
      html: `
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
          <h2 style="color: #333; border-bottom: 2px solid #4f46e5; padding-bottom: 10px;">
            New Contact Form Submission
          </h2>
          
          <div style="margin: 20px 0;">
            <h3 style="color: #4f46e5; margin-bottom: 5px;">From:</h3>
            <p style="margin: 0; padding: 10px; background-color: #f8fafc; border-radius: 5px;">
              <strong>Name:</strong> ${name}<br>
              <strong>Email:</strong> ${email}
            </p>
          </div>

          <div style="margin: 20px 0;">
            <h3 style="color: #4f46e5; margin-bottom: 5px;">Subject:</h3>
            <p style="margin: 0; padding: 10px; background-color: #f8fafc; border-radius: 5px;">
              ${subject}
            </p>
          </div>

          <div style="margin: 20px 0;">
            <h3 style="color: #4f46e5; margin-bottom: 5px;">Message:</h3>
            <div style="margin: 0; padding: 15px; background-color: #f8fafc; border-radius: 5px; white-space: pre-wrap;">
              ${message}
            </div>
          </div>

          <div style="margin-top: 30px; padding: 15px; background-color: #e0f2fe; border-radius: 5px; border-left: 4px solid #4f46e5;">
            <p style="margin: 0; color: #666; font-size: 14px;">
              This email was sent from your portfolio website contact form.
            </p>
          </div>
        </div>
      `,
      replyTo: email,
    };

    // Verify the transporter configuration
    await transporter.verify();

    // Send email
    await transporter.sendMail(mailOptions);

    return NextResponse.json({
      success: true,
      message: "Thank you for your message! I'll get back to you soon.",
    });
  } catch (error) {
    console.error("Contact form error:", error);
    
    // More specific error handling
    if (error instanceof Error) {
      if (error.message.includes("Invalid login")) {
        return NextResponse.json(
          { error: "Email service authentication failed. Please contact the administrator." },
          { status: 500 }
        );
      }
      if (error.message.includes("No recipients")) {
        return NextResponse.json(
          { error: "Invalid recipient email address." },
          { status: 400 }
        );
      }
    }
    
    return NextResponse.json(
      { error: "Failed to send email. Please try again later or contact me directly at hieptran.jobs@gmail.com." },
      { status: 500 }
    );
  }
}
