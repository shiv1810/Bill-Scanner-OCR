// DOM Elements
const fileInput = document.getElementById("fileInput");
const extractBtn = document.getElementById("extractBtn");
const fileName = document.getElementById("fileName");
const filePreview = document.getElementById("filePreview");
const removeFileBtn = document.getElementById("removeFile");
const loadingIndicator = document.querySelector(".loading-indicator");
const step1 = document.getElementById("step1");
const step2 = document.getElementById("step2");
const invoiceImg = document.getElementById("invoiceImg");
const successModal = document.getElementById("successModal");
const closeModalBtn = document.getElementById("closeModalBtn");
const confirmBtn = document.getElementById("confirmBtn");

// Event Listeners
fileInput.addEventListener("change", handleFileSelection);
removeFileBtn.addEventListener("click", removeFile);
extractBtn.addEventListener("click", uploadImage);
confirmBtn.addEventListener("click", confirmInvoice); // Added this line - it was missing!

closeModalBtn.addEventListener("click", () => {
  successModal.classList.add("hidden");
  goBack();
});

// Handle file selection
function handleFileSelection(event) {
  const file = event.target.files[0];
  if (file) {
    fileName.textContent = file.name;
    filePreview.classList.remove("hidden");
    extractBtn.disabled = false;

    // Preview the image
    const reader = new FileReader();
    reader.onload = (e) => {
      invoiceImg.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }
}

// Remove selected file
function removeFile() {
  fileInput.value = "";
  filePreview.classList.add("hidden");
  extractBtn.disabled = true;
}

// Go back to step 1
function goBack() {
  step1.classList.remove("hidden");
  step2.classList.add("hidden");
}

// Upload image and extract invoice data
function uploadImage() {
  const file = fileInput.files[0];

  if (!file) {
    alert("Please select an invoice image.");
    return;
  }

  // Show loading indicator
  extractBtn.disabled = true;
  loadingIndicator.classList.remove("hidden");

  // Show image preview
  const reader = new FileReader();
  reader.onload = (e) => {
    invoiceImg.src = e.target.result;
  };
  reader.readAsDataURL(file);

  // Add document type and style parameters
  const formData = new FormData();
  formData.append("file", file);
  formData.append("doc_type", "invoice");
  formData.append("doc_style", "digital");

  // Display a console message about the request
  console.log("Sending request to extract invoice data...");

  fetch("http://127.0.0.1:5000/upload-invoice", {
    method: "POST",
    body: formData,
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      return response.json();
    })
    .then((data) => {
      // Hide loading indicator
      loadingIndicator.classList.add("hidden");
      extractBtn.disabled = false;

      // Log the response for debugging
      console.log("Response from server:", data);

      if (data.status === "success") {
        // Debug extracted data
        console.log("Extracted data:", data.extracted_data);

        // Get the invoice data from the response
        const invoiceNumber = data.extracted_data?.invoice_number || "";
        const invoiceDate = data.extracted_data?.invoice_date || "";
        const invoiceAmount = data.extracted_data?.invoice_amount || "";

        console.log("Populating fields with:", {
          invoiceNumber,
          invoiceDate,
          invoiceAmount,
        });

        // Auto-fill form fields safely
        document.getElementById("invoiceNumberInput").value = invoiceNumber;
        document.getElementById("invoiceDateInput").value = invoiceDate;
        document.getElementById("invoiceAmountInput").value = invoiceAmount;

        // Move to step 2
        step1.classList.add("hidden");
        step2.classList.remove("hidden");
      } else {
        // Handle error in the response
        console.error("Error in response:", data.error || "Unknown error");
        alert("Error extracting details: " + (data.error || "Unknown error"));
      }
    })
    .catch((error) => {
      // Hide loading indicator
      loadingIndicator.classList.add("hidden");
      extractBtn.disabled = false;
      console.error("Error:", error);
      alert("Connection error. Please try again.");
    });
}

// Confirm and save invoice data
function confirmInvoice() {
  // Get form values
  const invoiceNumber = document.getElementById("invoiceNumberInput").value;
  const invoiceDate = document.getElementById("invoiceDateInput").value;
  const invoiceAmount = document.getElementById("invoiceAmountInput").value;

  // Validate form inputs
  if (!invoiceNumber || !invoiceDate || !invoiceAmount) {
    alert("Please fill in all fields.");
    return;
  }

  // Prepare JSON payload
  const payload = {
    invoice_number: invoiceNumber,
    invoice_date: invoiceDate,
    invoice_amount: parseFloat(invoiceAmount), // Convert to number
  };

  console.log("Submitting invoice data:", payload);

  // Show loading state
  confirmBtn.disabled = true;
  confirmBtn.textContent = "Saving...";

  fetch("http://127.0.0.1:5000/confirm-invoice", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      return response.json();
    })
    .then((data) => {
      confirmBtn.disabled = false;
      confirmBtn.textContent = "Confirm & Save";

      console.log("Confirmation response:", data);

      if (data.status === "success") {
        // Show success modal
        successModal.classList.remove("hidden");
      } else {
        alert("Error saving invoice data: " + (data.error || "Unknown error"));
      }
    })
    .catch((error) => {
      confirmBtn.disabled = false;
      confirmBtn.textContent = "Confirm & Save";
      console.error("Error:", error);
      alert("Connection error. Please try again.");
    });
}
