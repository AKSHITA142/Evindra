import ScrollWorld from "@/components/scroll/ScrollWorld";

// The landing page is a scroll-scrubbed cinematic "fly through the pipeline" world.
// The engine (components/scroll/scrubEngine.js) builds its own DOM, topbar, section
// nav and CTA; ScrollWorld mounts it client-side with the Evidra scene config.
// The previous marketing sections still live in components/landing/* if we want to
// re-add them below the experience or on a separate route.
export default function LandingPage() {
  return (
    <main className="bg-bg text-text">
      <ScrollWorld />
    </main>
  );
}
