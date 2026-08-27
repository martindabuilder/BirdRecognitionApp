import { BrowserRouter, Routes, Route, useLocation } from "react-router-dom"

import MainSection from "./components/MainSection/MainSection/Main_Section.jsx"
import ResultsSection from "./components/PredictResultsSection/Results_Section.jsx"
import BirdList from "./components/BirdListSection/Bird_Section.jsx"
import InfoSection from "./components/InfoSection/Info_Section.jsx"

import TitleBar from "./components/Shared/TitleBar.jsx"
import AuroraLayer from "./components/Shared/AuroraLayer.jsx"

import "./index.css"


function AppWrapper(){
    const location = useLocation()
    const isResultsPage = location.pathname === "/results"

    return (
        <div className={`app ${isResultsPage ? "results-theme" : "main-theme"}`}>
            <TitleBar />
            <div className = "app-content">
                <AuroraLayer />
                <Routes>
                    <Route path = "/" element = {<MainSection />}/>
                    <Route path = "/results" element = {<ResultsSection />}/>
                    <Route path = "/birdlist" element = {<BirdList />}/>
                    <Route path = "/information" element = {<InfoSection />}/>
                </Routes>
            </div>
        </div>
    )
}


function App() {
    return (
        <>
            <BrowserRouter>
                <AppWrapper />
            </BrowserRouter>
        </>
    )
}

export default App