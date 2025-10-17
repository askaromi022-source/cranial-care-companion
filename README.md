# Welcome to NeuroAI Pro: AI-Powered Brain Tumor Detection System

## Project info

**URL**: http://localhost:8080/

## How can I edit this code?

There are several ways of editing your application.

**Use Locally with Python & Node.js

You can run both the **backend (FastAPI)** and the **frontend (React)** locally.

**Use your preferred IDE**

If you want to work locally using your own IDE, you can clone this repo and push changes. Pushed changes will also be reflected in Lovable.

The only requirement is having Node.js & npm installed - [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating)

Follow these steps:

```sh
# Step 1: Clone the repository using the project's Git URL.
git clone <YOUR_GIT_URL>

# Step 2: Navigate to the project directory.
cd <YOUR_PROJECT_NAME>

# Step 3: Install the necessary dependencies.
npm i

# Step 4: Start the development server with auto-reloading and an instant preview.
npm run dev
```
#### Backend Setup
```bash
# Step 1: Navigate to backend folder
cd backend

# Step 2: Install Python dependencies
pip install -r requirements.txt

# Step 3: Run the FastAPI server
uvicorn app:app --reload

**Edit a file directly in GitHub**

- Navigate to the desired file(s).
- Click the "Edit" button (pencil icon) at the top right of the file view.
- Make your changes and commit the changes.

**Use GitHub Codespaces**

- Navigate to the main page of your repository.
- Click on the "Code" button (green button) near the top right.
- Select the "Codespaces" tab.
- Click on "New codespace" to launch a new Codespace environment.
- Edit files directly within the Codespace and commit and push your changes once you're done.

## What technologies are used for this project?

This project is built with:

- Vite
- TypeScript
- React
- shadcn-ui
- Tailwind CSS
- FastAPI

**Usage

Run the FastAPI backend.

Launch the React frontend.

Upload an MRI scan (.nii or .nii.gz).

See:

Original MRI slices

Tumor segmentation output

Confidence metrics
