// A function to open new issue in GitHub based on {{feedback_url}}.
// Activated when you click the "Send Feedback" button in the footer.
function openGitHubIssue() {
    var baseUrl = document.body.getAttribute('data-feedback-url');
    if (!baseUrl) {
        console.error('Feedback URL not found');
        return;
    }
    var pageUrl = encodeURIComponent(window.location.href);
    var pageTitle = document.querySelector('h1')?.textContent || 'Page';
    var issueTitle = `Feedback about ${pageTitle}`;
    var issueBody = `There is the following issue on this page: ${pageUrl}`;
    var feedbackUrl = `${baseUrl}/issues/new?title=${issueTitle}&body=${issueBody}&labels=module:%20docs`;
    window.open(feedbackUrl, '_blank');
}
