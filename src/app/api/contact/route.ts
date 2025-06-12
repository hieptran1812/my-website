import { NextRequest, NextResponse } from "next/server";
import nodemailer from "nodemailer";

// Simple in-memory rate limiting (in production, use Redis or database)
const rateLimit = new Map<string, { count: number; resetTime: number }>();

function checkRateLimit(ip: string): { allowed: boolean; retryAfter?: number } {
  const now = Date.now();
  const windowMs = 15 * 60 * 1000; // 15 minutes
  const maxRequests = 5; // 5 requests per 15 minutes

  const record = rateLimit.get(ip);

  if (!record || now > record.resetTime) {
    rateLimit.set(ip, { count: 1, resetTime: now + windowMs });
    return { allowed: true };
  }

  if (record.count >= maxRequests) {
    const retryAfter = Math.ceil((record.resetTime - now) / 1000);
    return { allowed: false, retryAfter };
  }

  record.count++;
  return { allowed: true };
}

export async function POST(request: NextRequest) {
  // Get client IP for rate limiting
  const ip =
    request.headers.get("x-forwarded-for") ||
    request.headers.get("x-real-ip") ||
    request.headers.get("cf-connecting-ip") ||
    "unknown";

  // Check rate limit
  const rateLimitResult = checkRateLimit(ip);
  if (!rateLimitResult.allowed) {
    return NextResponse.json(
      {
        error: `Too many requests. Please try again in ${rateLimitResult.retryAfter} seconds.`,
      },
      {
        status: 429,
        headers: {
          "Retry-After": rateLimitResult.retryAfter?.toString() || "900",
        },
      }
    );
  }

  try {
    const body = await request.json();
    const { name, email, subject, message } = body;

    // Enhanced validation
    if (
      !name?.trim() ||
      !email?.trim() ||
      !subject?.trim() ||
      !message?.trim()
    ) {
      return NextResponse.json(
        { error: "All fields are required and cannot be empty" },
        { status: 400 }
      );
    }

    // Enhanced email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email.trim())) {
      return NextResponse.json(
        { error: "Please provide a valid email address" },
        { status: 400 }
      );
    }

    // Length validation
    if (name.trim().length > 100) {
      return NextResponse.json(
        { error: "Name must be less than 100 characters" },
        { status: 400 }
      );
    }

    if (subject.trim().length > 200) {
      return NextResponse.json(
        { error: "Subject must be less than 200 characters" },
        { status: 400 }
      );
    }

    if (message.trim().length > 5000) {
      return NextResponse.json(
        { error: "Message must be less than 5000 characters" },
        { status: 400 }
      );
    }

    // Check environment variables
    if (!process.env.GMAIL_USER || !process.env.GMAIL_APP_PASSWORD) {
      console.error("Missing Gmail configuration in environment variables");
      return NextResponse.json(
        {
          error:
            "Email service is not configured properly. Please contact the administrator.",
        },
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

    // Email content with enhanced template
    const mailOptions = {
      from: process.env.GMAIL_USER,
      to: "hieptran.jobs@gmail.com",
      subject: `🔔 Portfolio Contact: ${subject}`,
      html: `
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <title>New Contact Form Submission</title>
        </head>
        <body style="margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f8fafc;">
          <div style="max-width: 600px; margin: 0 auto; background-color: #ffffff; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);">
            <!-- Header -->
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; text-align: center;">
              <h1 style="color: #ffffff; margin: 0; font-size: 24px; font-weight: 600;">
                📧 New Portfolio Message
              </h1>
              <p style="color: #e2e8f0; margin: 8px 0 0 0; font-size: 14px;">
                Received ${new Date().toLocaleString("en-US", {
                  weekday: "long",
                  year: "numeric",
                  month: "long",
                  day: "numeric",
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </p>
            </div>
            
            <!-- Content -->
            <div style="padding: 30px;">
              <!-- Sender Info -->
              <div style="background-color: #f8fafc; border-radius: 12px; padding: 20px; margin-bottom: 24px; border-left: 4px solid #667eea;">
                <h2 style="color: #1e293b; margin: 0 0 16px 0; font-size: 18px; display: flex; align-items: center;">
                  👤 Sender Information
                </h2>
                <div style="display: grid; gap: 12px;">
                  <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-weight: 600; color: #475569; min-width: 60px;">Name:</span>
                    <span style="color: #1e293b; background-color: #ffffff; padding: 6px 12px; border-radius: 6px; border: 1px solid #e2e8f0;">${name}</span>
                  </div>
                  <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-weight: 600; color: #475569; min-width: 60px;">Email:</span>
                    <a href="mailto:${email}" style="color: #667eea; text-decoration: none; background-color: #ffffff; padding: 6px 12px; border-radius: 6px; border: 1px solid #e2e8f0;">${email}</a>
                  </div>
                </div>
              </div>

              <!-- Subject -->
              <div style="background-color: #fef3c7; border-radius: 12px; padding: 20px; margin-bottom: 24px; border-left: 4px solid #f59e0b;">
                <h2 style="color: #92400e; margin: 0 0 12px 0; font-size: 18px; display: flex; align-items: center;">
                  📝 Subject
                </h2>
                <p style="color: #78350f; margin: 0; font-size: 16px; font-weight: 500; background-color: #fffbeb; padding: 12px; border-radius: 8px; border: 1px solid #fed7aa;">
                  ${subject}
                </p>
              </div>

              <!-- Message -->
              <div style="background-color: #ecfdf5; border-radius: 12px; padding: 20px; margin-bottom: 24px; border-left: 4px solid #10b981;">
                <h2 style="color: #065f46; margin: 0 0 16px 0; font-size: 18px; display: flex; align-items: center;">
                  💬 Message
                </h2>
                <div style="background-color: #f0fdf4; border-radius: 8px; padding: 16px; border: 1px solid #bbf7d0;">
                  <p style="color: #064e3b; margin: 0; line-height: 1.6; white-space: pre-wrap; font-size: 15px;">
                    ${message}
                  </p>
                </div>
              </div>

              <!-- Quick Actions -->
              <div style="background-color: #eff6ff; border-radius: 12px; padding: 20px; text-align: center; border-left: 4px solid #3b82f6;">
                <h3 style="color: #1d4ed8; margin: 0 0 16px 0; font-size: 16px;">
                  🚀 Quick Actions
                </h3>
                <div style="display: flex; gap: 12px; justify-content: center; flex-wrap: wrap;">
                  <a href="mailto:${email}?subject=Re: ${encodeURIComponent(
        subject
      )}" style="background-color: #3b82f6; color: white; padding: 10px 20px; border-radius: 6px; text-decoration: none; font-weight: 500; display: inline-flex; align-items: center; gap: 6px;">
                    ↩️ Reply
                  </a>
                  <a href="mailto:${email}" style="background-color: #10b981; color: white; padding: 10px 20px; border-radius: 6px; text-decoration: none; font-weight: 500; display: inline-flex; align-items: center; gap: 6px;">
                    📧 New Email
                  </a>
                </div>
              </div>
            </div>
            
            <!-- Footer -->
            <div style="background-color: #f1f5f9; padding: 20px; text-align: center; border-top: 1px solid #e2e8f0;">
              <p style="color: #64748b; margin: 0; font-size: 13px;">
                📍 This email was sent from your portfolio website contact form.<br>
                🔒 Sender IP: ${ip} | ⏰ Timestamp: ${new Date().toISOString()}
              </p>
            </div>
          </div>
        </body>
        </html>
      `,
      replyTo: email,
    };

    // Verify the transporter configuration
    await transporter.verify();

    // Send email to me
    await transporter.sendMail(mailOptions);

    // Send confirmation email to the user
    const confirmationEmail = {
      from: process.env.GMAIL_USER,
      to: email,
      subject: "✅ Message Received - Thank You for Contacting Me!",
      html: `
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <title>Thank You for Your Message</title>
        </head>
        <body style="margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f8fafc;">
          <div style="max-width: 600px; margin: 0 auto; background-color: #ffffff; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);">
            <!-- Header -->
            <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); padding: 30px; text-align: center;">
              <h1 style="color: #ffffff; margin: 0; font-size: 28px; font-weight: 600;">
                ✨ Thank You!
              </h1>
              <p style="color: #d1fae5; margin: 12px 0 0 0; font-size: 16px;">
                Your message has been received successfully
              </p>
            </div>
            
            <!-- Content -->
            <div style="padding: 40px 30px;">
              <p style="color: #1f2937; font-size: 18px; line-height: 1.6; margin: 0 0 24px 0;">
                Hi <strong>${name}</strong>,
              </p>
              
              <p style="color: #374151; font-size: 16px; line-height: 1.6; margin: 0 0 20px 0;">
                Thank you for reaching out! I've received your message about "<strong>${subject}</strong>" and I'm excited to connect with you.
              </p>
              
              <div style="background-color: #f0f9ff; border-radius: 12px; padding: 24px; margin: 24px 0; border-left: 4px solid #0ea5e9;">
                <h3 style="color: #0c4a6e; margin: 0 0 16px 0; font-size: 18px; display: flex; align-items: center;">
                  📋 Message Summary
                </h3>
                <div style="background-color: #ffffff; border-radius: 8px; padding: 16px; border: 1px solid #e0f2fe;">
                  <p style="color: #374151; margin: 0; font-size: 14px; line-height: 1.5;">
                    <strong>Subject:</strong> ${subject}<br>
                    <strong>Sent:</strong> ${new Date().toLocaleString(
                      "en-US",
                      {
                        weekday: "long",
                        year: "numeric",
                        month: "long",
                        day: "numeric",
                        hour: "2-digit",
                        minute: "2-digit",
                      }
                    )}<br>
                    <strong>Your Email:</strong> ${email}
                  </p>
                </div>
              </div>
              
              <p style="color: #374151; font-size: 16px; line-height: 1.6; margin: 24px 0;">
                ⏰ <strong>What's Next?</strong><br>
                I typically respond to messages within 24 hours. I'll review your inquiry and get back to you with a thoughtful response.
              </p>
              
              <div style="background-color: #fef3c7; border-radius: 12px; padding: 20px; margin: 24px 0; border-left: 4px solid #f59e0b;">
                <p style="color: #92400e; margin: 0; font-size: 14px; line-height: 1.5;">
                  <strong>💡 In the meantime:</strong><br>
                  Feel free to check out my latest projects and blog posts on my website. You can also connect with me on <a href="https://www.linkedin.com/in/hieptran01" style="color: #0ea5e9; text-decoration: none;">LinkedIn</a> or <a href="https://github.com/hieptran1812" style="color: #0ea5e9; text-decoration: none;">GitHub</a>.
                </p>
              </div>
              
              <p style="color: #374151; font-size: 16px; line-height: 1.6; margin: 24px 0 0 0;">
                Best regards,<br>
                <strong style="color: #1f2937;">Hiep Tran</strong><br>
                <span style="color: #6b7280; font-size: 14px;">Software Developer & AI Enthusiast</span>
              </p>
            </div>
            
            <!-- Footer -->
            <div style="background-color: #f1f5f9; padding: 20px; text-align: center; border-top: 1px solid #e2e8f0;">
              <p style="color: #64748b; margin: 0 0 12px 0; font-size: 13px;">
                📧 This is an automated confirmation email
              </p>
              <div style="display: flex; justify-content: center; gap: 16px; flex-wrap: wrap;">
                <a href="https://www.linkedin.com/in/hieptran01" style="color: #0ea5e9; text-decoration: none; font-size: 12px;">LinkedIn</a>
                <a href="https://github.com/hieptran1812" style="color: #0ea5e9; text-decoration: none; font-size: 12px;">GitHub</a>
                <a href="mailto:hieptran.jobs@gmail.com" style="color: #0ea5e9; text-decoration: none; font-size: 12px;">Direct Email</a>
              </div>
            </div>
          </div>
        </body>
        </html>
      `,
    };

    // Send confirmation email (don't fail if this fails)
    try {
      await transporter.sendMail(confirmationEmail);
    } catch (confirmationError) {
      console.warn("Failed to send confirmation email:", confirmationError);
      // Continue without failing the main request
    }

    return NextResponse.json({
      success: true,
      message:
        "🎉 Thank you for your message! I'll get back to you within 24 hours. Stay tuned :D",
    });
  } catch (error) {
    console.error("Contact form error:", error);

    // More specific error handling
    if (error instanceof Error) {
      if (error.message.includes("Invalid login")) {
        return NextResponse.json(
          {
            error:
              "Email service authentication failed. Please contact the administrator.",
          },
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
      {
        error:
          "Failed to send email. Please try again later or contact me directly at hieptran.jobs@gmail.com.",
      },
      { status: 500 }
    );
  }
}
