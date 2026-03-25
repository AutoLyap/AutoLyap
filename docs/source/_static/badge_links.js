(() => {
  function updateBadgeLinks() {
    const badgeImages = document.querySelectorAll('a[href] > img[src*="img.shields.io"]');

    badgeImages.forEach((image) => {
      const link = image.closest("a");
      if (!(link instanceof HTMLAnchorElement)) {
        return;
      }

      link.target = "_blank";
      link.rel = "noopener noreferrer";
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", updateBadgeLinks, { once: true });
    return;
  }

  updateBadgeLinks();
})();
