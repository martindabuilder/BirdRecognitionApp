import { BrowserRouter, Routes, Route } from "react-router-dom"

import "./index.css"

import MainSection from "./components/MainSection/MainSection/Main_Section.jsx"
import ResultsSection from "./components/PredictResultsSection/Results_Section.jsx"
import BirdList from "./components/BirdListSection/Bird_Section.jsx"
import InfoSection from "./components/InfoSection/Info_Section.jsx"

function App() {
    return (
        <BrowserRouter>
            <Routes>
                <Route path = "/" element = {<MainSection />}/>
                <Route path = "/results" element = {<ResultsSection />}/>
                <Route path = "/bird/list" element = {<BirdList />}/>
                <Route path = "/information" element = {<InfoSection />}/>
            </Routes>
        </BrowserRouter>
    )
}

export default App