# TimescaleDB Setup for Postgres.app (macOS)

## Current Setup
- Postgres.app with PostgreSQL 17
- Location: `/Applications/Postgres.app/Contents/Versions/17/`
- Extension directory: `/Applications/Postgres.app/Contents/Versions/17/lib/postgresql/`

## Installation Method

Since Postgres.app uses its own bundled PostgreSQL, we'll install TimescaleDB via Homebrew and then copy it to Postgres.app.

### Step 1: Install TimescaleDB via Homebrew

```bash
# Install TimescaleDB (this installs it for Homebrew PostgreSQL)
brew install timescaledb
```

This will install TimescaleDB for PostgreSQL 17 (matching your Postgres.app version).

### Step 2: Locate TimescaleDB Files

After installation, find where Homebrew installed TimescaleDB:

```bash
# Find TimescaleDB library files
find /opt/homebrew -name "*timescaledb*" -type f 2>/dev/null

# Or check the typical location
ls -la /opt/homebrew/lib/postgresql@17/
```

The key files you need are:
- `timescaledb-*.so` (or `.dylib` on macOS) - the main extension library
- SQL files in the extension directory

### Step 3: Copy TimescaleDB to Postgres.app

```bash
# Create a backup first (optional but recommended)
cp -r /Applications/Postgres.app/Contents/Versions/17/lib/postgresql /Applications/Postgres.app/Contents/Versions/17/lib/postgresql.backup

# Copy TimescaleDB library to Postgres.app
# Adjust the source path based on where Homebrew installed it
sudo cp /opt/homebrew/lib/postgresql@17/timescaledb-*.so \
       /Applications/Postgres.app/Contents/Versions/17/lib/postgresql/

# Copy SQL files (if in a separate share directory)
# Find the share directory:
find /opt/homebrew -path "*/share/postgresql@17/extension/timescaledb*" 2>/dev/null

# Copy SQL files:
sudo cp -r /opt/homebrew/share/postgresql@17/extension/timescaledb* \
           /Applications/Postgres.app/Contents/Versions/17/share/postgresql/extension/
```

**Note:** You may need to adjust paths based on your Homebrew installation location:
- Apple Silicon Mac: `/opt/homebrew/`
- Intel Mac: `/usr/local/`

### Step 4: Configure Postgres.app to Load TimescaleDB

Find your Postgres.app data directory and configuration file:

```bash
# Find postgresql.conf (usually in Application Support)
find ~/Library/Application\ Support/Postgres -name "postgresql.conf" 2>/dev/null

# Or check for var-17 directory (for PostgreSQL 17)
ls -la ~/Library/Application\ Support/Postgres/
```

Edit `postgresql.conf` and add/modify:

```conf
shared_preload_libraries = 'timescaledb'
```

**Important:** If `shared_preload_libraries` already has values, add TimescaleDB to the list:
```conf
shared_preload_libraries = 'timescaledb,other_extension'
```

### Step 5: Restart Postgres.app

1. Quit Postgres.app completely (right-click the menu bar icon → Quit)
2. Restart Postgres.app from Applications

### Step 6: Verify Installation

Connect to your database using Postgres.app's psql or your connection string:

```bash
# Using Postgres.app's psql
/Applications/Postgres.app/Contents/Versions/17/bin/psql -d gluroo_datasets

# Or if Postgres.app's bin is in your PATH
psql -d gluroo_datasets
```

Then in psql:

```sql
-- Create the extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Verify it's installed
SELECT default_version, installed_version
FROM pg_available_extensions
WHERE name = 'timescaledb';

-- Check if it's loaded
\dx timescaledb
```

### Step 7: Test with Your Schema

Run your `schema.sql` script:

```bash
psql -d gluroo_datasets -f src/data/diabetes_datasets/gluroo/schema.sql
```

The script should now successfully:
- Create the TimescaleDB extension
- Convert `readings` and `messages` tables to hypertables
- Add space partitioning by `gid`

## Alternative: Build from Source (If Copy Method Doesn't Work)

If copying files doesn't work, you can build TimescaleDB specifically for Postgres.app:

### Prerequisites

```bash
brew install cmake openssl
```

### Build Steps

```bash
# Clone TimescaleDB
git clone https://github.com/timescale/timescaledb.git
cd timescaledb

# Bootstrap with Postgres.app's PostgreSQL path
./bootstrap \
  -DREGRESS_CHECKS=OFF \
  -DPG_CONFIG=/Applications/Postgres.app/Contents/Versions/17/bin/pg_config

# Build
cd build
make

# Install to Postgres.app
sudo make install DESTDIR=/Applications/Postgres.app/Contents/Versions/17
```

Then follow Steps 4-7 above.

## Troubleshooting

### Issue: "could not access file 'timescaledb'"

**Solution:**
1. Verify the library file exists:
   ```bash
   ls -la /Applications/Postgres.app/Contents/Versions/17/lib/postgresql/timescaledb*
   ```

2. Check file permissions:
   ```bash
   sudo chmod 755 /Applications/Postgres.app/Contents/Versions/17/lib/postgresql/timescaledb-*.so
   ```

3. Verify `shared_preload_libraries` is set correctly:
   ```sql
   SHOW shared_preload_libraries;
   ```

### Issue: Extension not found

**Solution:**
1. Check if SQL files are in the right place:
   ```bash
   ls -la /Applications/Postgres.app/Contents/Versions/17/share/postgresql/extension/timescaledb*
   ```

2. Check available extensions:
   ```sql
   SELECT * FROM pg_available_extensions WHERE name LIKE '%timescale%';
   ```

### Issue: Version mismatch

**Solution:**
Make sure TimescaleDB version matches PostgreSQL 17. Check compatibility:
```bash
# Check TimescaleDB version
/opt/homebrew/bin/timescaledb --version

# Check PostgreSQL version
/Applications/Postgres.app/Contents/Versions/17/bin/postgres --version
```

### Issue: Permission denied

**Solution:**
You may need to adjust permissions on Postgres.app files. Be careful with this:
```bash
# Only if necessary - adjust ownership
sudo chown -R $(whoami) /Applications/Postgres.app/Contents/Versions/17/lib/postgresql/
```

## Verification Commands

After installation, verify everything works:

```sql
-- 1. Check extension is installed
\dx

-- 2. Check TimescaleDB version
SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';

-- 3. After running schema.sql, check hypertables
SELECT * FROM timescaledb_information.hypertables;

-- 4. Check chunk information
SELECT * FROM timescaledb_information.chunks LIMIT 5;
```

## Quick Reference

**Postgres.app Paths:**
- Binary: `/Applications/Postgres.app/Contents/Versions/17/bin/`
- Libraries: `/Applications/Postgres.app/Contents/Versions/17/lib/postgresql/`
- Extensions: `/Applications/Postgres.app/Contents/Versions/17/share/postgresql/extension/`
- Config: `~/Library/Application Support/Postgres/var-17/postgresql.conf`

**Useful Commands:**
```bash
# Connect to database
/Applications/Postgres.app/Contents/Versions/17/bin/psql -d gluroo_datasets

# Check PostgreSQL version
/Applications/Postgres.app/Contents/Versions/17/bin/postgres --version

# Find TimescaleDB files
find /opt/homebrew -name "*timescaledb*" 2>/dev/null
```

## Next Steps

1. Install TimescaleDB via Homebrew
2. Copy files to Postgres.app
3. Configure `postgresql.conf`
4. Restart Postgres.app
5. Run your `schema.sql` script
6. Verify hypertables were created

Your `schema.sql` will automatically use TimescaleDB features if available, or fall back to regular PostgreSQL tables if not.
