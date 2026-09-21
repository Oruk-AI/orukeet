import Foundation
import XCTest

final class OrukeetExampleUITests: XCTestCase {
    @MainActor
    func testInstallPrepareAndRepeatedFixtureTranscription() throws {
        continueAfterFailure = false
        let environment = ProcessInfo.processInfo.environment
        let archive = try XCTUnwrap(environment["ORUKEET_EXAMPLE_TEST_ARCHIVE"],
            "Set TEST_RUNNER_ORUKEET_EXAMPLE_TEST_ARCHIVE to the host ZIP path used by the model smoke test.")
        let fixture = try XCTUnwrap(environment["ORUKEET_EXAMPLE_TEST_FIXTURE"],
            "Set TEST_RUNNER_ORUKEET_EXAMPLE_TEST_FIXTURE to demos/fixtures/jfk.wav.")
        let reportPath = try XCTUnwrap(environment["ORUKEET_EXAMPLE_TEST_REPORT"],
            "Set TEST_RUNNER_ORUKEET_EXAMPLE_TEST_REPORT to the output JSON path on the CI runner.")
        // Use a host scratch path beside the low-level smoke test's archive,
        // outside app containers and uploaded evidence. The app downloads its own
        // archive through the public HTTPS installer; this path is not imported.
        // Only this new UUID child is deleted.
        let installRoot = URL(fileURLWithPath: archive).deletingLastPathComponent()
            .appendingPathComponent("orukeet-ui-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: installRoot) }
        let app = XCUIApplication()
        app.launchArguments = [
            "--orukeet-fixture", fixture,
            "--orukeet-test-root", installRoot.path,
        ]
        app.launch()
        let install = app.buttons["installButton"]
        XCTAssertTrue(install.waitForExistence(timeout: 30))
        waitUntilEnabled(install, timeout: 30)
        install.tap()
        let record = app.buttons["recordButton"]
        waitUntilEnabled(record, timeout: 600)
        XCTAssertFalse(app.staticTexts["errorMessage"].exists)

        let fixtureButton = app.buttons["fixtureButton"]
        XCTAssertTrue(fixtureButton.exists)
        fixtureButton.tap()
        waitForCompletedCount(1, app: app)
        let transcript = app.staticTexts["transcriptText"]
        XCTAssertTrue(transcript.exists)
        let first = transcript.label
        XCTAssertTrue(first.lowercased().contains("country"), "Unexpected JFK transcript: \(first)")
        XCTAssertFalse(app.staticTexts["errorMessage"].exists)

        fixtureButton.tap()
        waitForCompletedCount(2, app: app)
        XCTAssertEqual(transcript.label, first, "A second recording must not inherit decoder state.")
        let attachment = XCTAttachment(string: first)
        attachment.name = "Orukeet INT8 app transcript"
        attachment.lifetime = .keepAlways
        add(attachment)
        app.terminate()

        // Cold relaunch has no archive or network-install action. The same
        // authenticated cache must be discovered and prepared automatically.
        app.launchArguments = [
            "--orukeet-fixture", fixture,
            "--orukeet-test-root", installRoot.path,
        ]
        app.launch()
        waitUntilEnabled(app.buttons["recordButton"], timeout: 180)
        XCTAssertFalse(app.buttons["installButton"].exists)
        XCTAssertFalse(app.staticTexts["errorMessage"].exists)
        app.buttons["fixtureButton"].tap()
        waitForCompletedCount(1, app: app)
        let relaunched = app.staticTexts["transcriptText"].label
        XCTAssertEqual(relaunched, first, "Offline relaunch must reuse the installed model.")
        app.terminate()

        let report: [String: Any] = [
            "scope": "iOS Simulator example app end-to-end",
            "installation_source": "pinned HTTPS download",
            "all_assertions_passed": true,
            "install_and_prepare_passed": true,
            "repeat_identical": true,
            "offline_relaunch_passed": true,
            "fixture_path": fixture,
            "transcript": first,
            "offline_relaunch_transcript": relaunched,
            "recorded_at": ISO8601DateFormatter().string(from: Date()),
            "limitations": [
                "Uses a real audio fixture instead of microphone capture",
                "Simulator results do not measure physical iPhone performance",
            ],
        ]
        let reportURL = URL(fileURLWithPath: reportPath)
        try FileManager.default.createDirectory(at: reportURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: reportURL, options: .atomic)
    }

    @MainActor
    private func waitUntilEnabled(_ element: XCUIElement, timeout: TimeInterval) {
        let ready = XCTNSPredicateExpectation(predicate: NSPredicate(format: "exists == true AND enabled == true"), object: element)
        XCTAssertEqual(XCTWaiter.wait(for: [ready], timeout: timeout), .completed)
    }

    @MainActor
    private func waitForCompletedCount(_ count: Int, app: XCUIApplication) {
        let completed = XCTNSPredicateExpectation(
            predicate: NSPredicate(format: "label == %@", "Completed recordings: \(count)"),
            object: app.staticTexts["completedRecordings"])
        XCTAssertEqual(XCTWaiter.wait(for: [completed], timeout: 180), .completed)
    }
}
