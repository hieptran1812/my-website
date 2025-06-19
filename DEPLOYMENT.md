# Deployment Guide

## Vercel Deployment

This project is configured to deploy on Vercel with the following setup:

### Prerequisites

1. **Environment Variables**: Set up the following environment variables in your Vercel dashboard:
   - `DATABASE_URL`: For production, consider using a cloud database like PlanetScale, Supabase, or Neon
   - `GMAIL_USER`: Your Gmail address for the contact form
   - `GMAIL_APP_PASSWORD`: Gmail app password for sending emails
   - `NEXT_PUBLIC_SITE_URL`: Your production domain

### Deployment Steps

1. **Push your changes to GitHub**:

   ```bash
   git add .
   git commit -m "Fix Prisma build issues for Vercel deployment"
   git push origin main
   ```

2. **Set up environment variables in Vercel**:

   - Go to your Vercel project dashboard
   - Navigate to Settings → Environment Variables
   - Add the required variables (see Prerequisites section)

3. **Deploy**:
   - Vercel will automatically deploy from your main branch
   - The build process will now properly generate Prisma Client
   - Monitor the build logs for any issues

### Build Configuration

The project includes:

- **Automatic Prisma Generation**: `prisma generate` runs before every build
- **Post-install hook**: Ensures Prisma client is generated after dependency installation
- **Vercel.json**: Optimized configuration for serverless functions

### Database Setup

#### For Development

```bash
npm install
npx prisma migrate dev
npx prisma generate
```

#### For Production

1. **Set up a cloud database** (required for production):

   **Option A: Neon (Recommended)**

   ```bash
   # 1. Create account at https://neon.tech
   # 2. Create a new project
   # 3. Copy the connection string
   # 4. Set in Vercel: DATABASE_URL="postgresql://..."
   ```

   **Option B: Supabase**

   ```bash
   # 1. Create account at https://supabase.com
   # 2. Create new project
   # 3. Go to Settings → Database
   # 4. Copy connection string
   # 5. Set in Vercel: DATABASE_URL="postgresql://..."
   ```

   **Option C: PlanetScale**

   ```bash
   # 1. Create account at https://planetscale.com
   # 2. Create new database
   # 3. Get connection string
   # 4. Set in Vercel: DATABASE_URL="mysql://..."
   ```

2. **Deploy the schema**:

   ```bash
   # After setting DATABASE_URL in Vercel environment variables
   npx prisma db push
   ```

3. **Important**: SQLite (`file:./dev.db`) will NOT work on Vercel. You must use a cloud database for production.

### Common Issues & Solutions

#### Issue: "Prisma has detected that this project was built on Vercel"

**Solution**: This is automatically handled by:

- `postinstall` script in package.json that runs `prisma generate`
- `vercel-build` script for explicit Vercel builds
- Updated Prisma configuration with proper binary targets
- Next.js webpack configuration for server-side externals

#### Issue: PrismaClientInitializationError during build

**Solution**:

- The project is configured with proper binary targets in `prisma/schema.prisma`
- Webpack configuration excludes Prisma from client bundle
- Build process ensures Prisma Client is generated before Next.js build

#### Issue: Database not accessible in production

**Solution**:

- Use a cloud database provider
- Ensure DATABASE_URL is properly configured
- Run `npx prisma db push` to sync schema

### Manual Deployment Steps

1. **Push to GitHub**: Ensure all changes are committed
2. **Check Environment Variables**: Verify all required env vars are set in Vercel
3. **Deploy**: Vercel will automatically trigger deployment on push
4. **Verify**: Check that all API routes are working correctly

### Performance Optimizations

- Static page generation for blog posts
- Image optimization with Next.js Image component
- Database query optimization with Prisma
- Caching headers for static assets
